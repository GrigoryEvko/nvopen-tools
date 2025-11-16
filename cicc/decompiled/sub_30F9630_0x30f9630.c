// Function: sub_30F9630
// Address: 0x30f9630
//
__int64 __fastcall sub_30F9630(__int64 a1, unsigned __int8 *a2)
{
  __int64 *v4; // r14
  __int64 *v5; // r15
  unsigned __int8 v6; // di
  __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // eax
  int v10; // eax
  unsigned int v11; // esi
  __int64 **v12; // rdx
  __int64 *v13; // r8
  __int64 *v14; // rdx
  unsigned int v15; // esi
  __int64 **v16; // rdx
  __int64 *v17; // rdi
  __int64 v18; // rax
  int v19; // edi
  __int64 v20; // rbx
  int v21; // r9d
  __int64 v22; // rsi
  int v23; // edx
  unsigned int v24; // ecx
  unsigned __int8 v25; // al
  __int64 *v26; // rax
  unsigned __int8 *v27; // r14
  __int64 v28; // rdi
  unsigned __int8 v30; // cl
  int v31; // ecx
  __int64 v32; // rax
  int v33; // eax
  int v34; // edx
  int v35; // r9d
  int v36; // edx
  int v37; // r8d
  __int64 v38; // [rsp+8h] [rbp-88h]
  __m128i v39; // [rsp+10h] [rbp-80h] BYREF
  __int64 v40; // [rsp+20h] [rbp-70h]
  __int64 v41; // [rsp+28h] [rbp-68h]
  __int64 v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+38h] [rbp-58h]
  __int64 v44; // [rsp+40h] [rbp-50h]
  __int64 v45; // [rsp+48h] [rbp-48h]
  __int16 v46; // [rsp+50h] [rbp-40h]

  v4 = (__int64 *)*((_QWORD *)a2 - 8);
  v5 = (__int64 *)*((_QWORD *)a2 - 4);
  v6 = *(_BYTE *)v5;
  if ( *(_BYTE *)v4 <= 0x15u )
  {
    if ( v6 <= 0x15u )
      goto LABEL_10;
    v32 = *(_QWORD *)(a1 + 40);
    v8 = *(_QWORD *)(v32 + 8);
    v33 = *(_DWORD *)(v32 + 24);
    if ( !v33 )
      goto LABEL_10;
    v10 = v33 - 1;
    goto LABEL_7;
  }
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(v7 + 8);
  v9 = *(_DWORD *)(v7 + 24);
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v12 = (__int64 **)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v4 == *v12 )
    {
LABEL_4:
      v14 = v12[1];
      if ( v14 )
        v4 = v14;
    }
    else
    {
      v34 = 1;
      while ( v13 != (__int64 *)-4096LL )
      {
        v35 = v34 + 1;
        v11 = v10 & (v34 + v11);
        v12 = (__int64 **)(v8 + 16LL * v11);
        v13 = *v12;
        if ( v4 == *v12 )
          goto LABEL_4;
        v34 = v35;
      }
    }
    if ( v6 > 0x15u )
    {
LABEL_7:
      v15 = v10 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v16 = (__int64 **)(v8 + 16LL * v15);
      v17 = *v16;
      if ( v5 == *v16 )
      {
LABEL_8:
        if ( v16[1] )
          v5 = v16[1];
      }
      else
      {
        v36 = 1;
        while ( v17 != (__int64 *)-4096LL )
        {
          v37 = v36 + 1;
          v15 = v10 & (v36 + v15);
          v16 = (__int64 **)(v8 + 16LL * v15);
          v17 = *v16;
          if ( v5 == *v16 )
            goto LABEL_8;
          v36 = v37;
        }
      }
    }
  }
LABEL_10:
  v18 = sub_B43CC0((__int64)a2);
  v19 = *a2;
  v20 = v18;
  v21 = v19 - 29;
  if ( (unsigned __int8)v19 > 0x1Cu )
  {
    switch ( *a2 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_28;
      case 'T':
      case 'U':
      case 'V':
        v22 = *((_QWORD *)a2 + 1);
        v23 = *(unsigned __int8 *)(v22 + 8);
        v24 = v23 - 17;
        v25 = *(_BYTE *)(v22 + 8);
        if ( (unsigned int)(v23 - 17) <= 1 )
          v25 = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
        if ( v25 <= 3u || v25 == 5 || (v25 & 0xFD) == 4 )
          goto LABEL_28;
        if ( (_BYTE)v23 == 15 )
        {
          if ( (*(_BYTE *)(v22 + 9) & 4) == 0 )
            break;
          v38 = *((_QWORD *)a2 + 1);
          if ( !sub_BCB420(v38) )
          {
            v21 = *a2 - 29;
            break;
          }
          v26 = *(__int64 **)(v38 + 16);
          v22 = *v26;
          v21 = *a2 - 29;
          v23 = *(unsigned __int8 *)(*v26 + 8);
          v24 = v23 - 17;
        }
        else if ( (_BYTE)v23 == 16 )
        {
          do
          {
            v22 = *(_QWORD *)(v22 + 24);
            LOBYTE(v23) = *(_BYTE *)(v22 + 8);
          }
          while ( (_BYTE)v23 == 16 );
          v24 = (unsigned __int8)v23 - 17;
        }
        if ( v24 <= 1 )
          LOBYTE(v23) = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
        if ( (unsigned __int8)v23 > 3u && (_BYTE)v23 != 5 && (v23 & 0xFD) != 4 )
          break;
LABEL_28:
        v30 = a2[1];
        v39 = (__m128i)(unsigned __int64)v20;
        v46 = 257;
        v31 = v30 >> 1;
        v40 = 0;
        v41 = 0;
        if ( v31 == 127 )
          LOBYTE(v31) = -1;
        v42 = 0;
        v43 = 0;
        v44 = 0;
        v45 = 0;
        v27 = sub_101E830(v21, v4, v5, v31, &v39);
        if ( !v27 )
          return sub_30F9620(a1, (__int64)a2);
        goto LABEL_27;
      default:
        break;
    }
  }
  v39 = (__m128i)(unsigned __int64)v20;
  v46 = 257;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v27 = sub_101E7C0(v21, v4, v5, &v39);
  if ( !v27 )
    return sub_30F9620(a1, (__int64)a2);
LABEL_27:
  v28 = *(_QWORD *)(a1 + 40);
  v39.m128i_i64[0] = (__int64)a2;
  *sub_FAA780(v28, v39.m128i_i64) = v27;
  return 1;
}
