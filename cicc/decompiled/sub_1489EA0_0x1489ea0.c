// Function: sub_1489EA0
// Address: 0x1489ea0
//
__int64 *__fastcall sub_1489EA0(__int64 *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 *v4; // r15
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 **v9; // rdx
  __int64 *v10; // r10
  int v12; // edx
  __int64 v13; // rsi
  __int64 v14; // rdx
  _QWORD *v15; // rbx
  __int64 *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  _QWORD *v28; // rbx
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  _QWORD *v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rbx
  int v35; // r11d
  __int64 v36; // rsi
  _QWORD *v37; // [rsp+8h] [rbp-98h]
  _QWORD *v38; // [rsp+8h] [rbp-98h]
  _QWORD *v39; // [rsp+8h] [rbp-98h]
  _QWORD *v40; // [rsp+8h] [rbp-98h]
  _QWORD *v41; // [rsp+8h] [rbp-98h]
  char v42; // [rsp+1Fh] [rbp-81h]
  char v43; // [rsp+1Fh] [rbp-81h]
  char v44; // [rsp+1Fh] [rbp-81h]
  char v45; // [rsp+1Fh] [rbp-81h]
  char v46; // [rsp+1Fh] [rbp-81h]
  __int64 v47; // [rsp+20h] [rbp-80h]
  __int64 v48; // [rsp+20h] [rbp-80h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  __int64 v51; // [rsp+20h] [rbp-80h]
  __int64 v52; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v53; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v54; // [rsp+40h] [rbp-60h] BYREF
  __int64 v55; // [rsp+48h] [rbp-58h]
  _QWORD v56[10]; // [rsp+50h] [rbp-50h] BYREF

  v4 = (__int64 *)a2;
  v6 = *((unsigned int *)a1 + 8);
  v52 = a2;
  if ( (_DWORD)v6 )
  {
    v7 = a1[2];
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 **)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 **)(v7 + 16 * v6) )
        return v9[1];
    }
    else
    {
      v12 = 1;
      while ( v10 != (__int64 *)-8LL )
      {
        v35 = v12 + 1;
        v8 = (v6 - 1) & (v12 + v8);
        v9 = (__int64 **)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_3;
        v12 = v35;
      }
    }
  }
  switch ( *((_WORD *)v4 + 12) )
  {
    case 0:
    case 0xB:
      break;
    case 1:
      v13 = sub_1489EA0(a1, v4[4]);
      if ( v13 != v4[4] )
        v4 = (__int64 *)sub_14835F0((_QWORD *)*a1, v13, v4[5], 0, a3, a4);
      break;
    case 2:
      v26 = sub_1489EA0(a1, v4[4]);
      if ( v26 != v4[4] )
        v4 = (__int64 *)sub_14747F0(*a1, v26, v4[5], 0);
      break;
    case 3:
      v30 = sub_1489EA0(a1, v4[4]);
      if ( v30 != v4[4] )
        v4 = (__int64 *)sub_147B0D0(*a1, v30, v4[5], 0);
      break;
    case 4:
      v54 = v56;
      v55 = 0x200000000LL;
      v27 = v4[4];
      v40 = (_QWORD *)(v27 + 8 * v4[5]);
      if ( (_QWORD *)v27 != v40 )
      {
        v45 = 0;
        v28 = (_QWORD *)v4[4];
        do
        {
          v50 = *v28;
          v53 = (__int64 *)sub_1489EA0(a1, *v28);
          sub_1458920((__int64)&v54, &v53);
          v16 = v54;
          ++v28;
          v45 |= v54[(unsigned int)v55 - 1] != v50;
        }
        while ( v40 != v28 );
        if ( v45 )
        {
          v29 = sub_147DD40(*a1, (__int64 *)&v54, 0, 0, a3, a4);
          v16 = v54;
          v4 = v29;
        }
        goto LABEL_35;
      }
      break;
    case 5:
      v54 = v56;
      v55 = 0x200000000LL;
      v31 = v4[4];
      v41 = (_QWORD *)(v31 + 8 * v4[5]);
      if ( (_QWORD *)v31 != v41 )
      {
        v46 = 0;
        v32 = (_QWORD *)v4[4];
        do
        {
          v51 = *v32;
          v53 = (__int64 *)sub_1489EA0(a1, *v32);
          sub_1458920((__int64)&v54, &v53);
          v16 = v54;
          ++v32;
          v46 |= v54[(unsigned int)v55 - 1] != v51;
        }
        while ( v41 != v32 );
        if ( v46 )
        {
          v33 = sub_147EE30((_QWORD *)*a1, &v54, 0, 0, a3, a4);
          v16 = v54;
          v4 = (__int64 *)v33;
        }
        goto LABEL_35;
      }
      break;
    case 6:
      v24 = sub_1489EA0(a1, v4[4]);
      v25 = sub_1489EA0(a1, v4[5]);
      if ( v24 != v4[4] || v25 != v4[5] )
        v4 = (__int64 *)sub_1483CF0((_QWORD *)*a1, v24, v25, a3, a4);
      break;
    case 7:
      v54 = v56;
      v55 = 0x200000000LL;
      v21 = v4[4];
      v39 = (_QWORD *)(v21 + 8 * v4[5]);
      if ( (_QWORD *)v21 != v39 )
      {
        v44 = 0;
        v22 = (_QWORD *)v4[4];
        do
        {
          v49 = *v22;
          v53 = (__int64 *)sub_1489EA0(a1, *v22);
          sub_1458920((__int64)&v54, &v53);
          v16 = v54;
          ++v22;
          v44 |= v54[(unsigned int)v55 - 1] != v49;
        }
        while ( v39 != v22 );
        if ( v44 )
        {
          v23 = sub_14785F0(*a1, &v54, v4[6], *((_WORD *)v4 + 13) & 7);
          v16 = v54;
          v4 = (__int64 *)v23;
        }
        goto LABEL_35;
      }
      break;
    case 8:
      v54 = v56;
      v55 = 0x200000000LL;
      v18 = v4[4];
      v38 = (_QWORD *)(v18 + 8 * v4[5]);
      if ( (_QWORD *)v18 != v38 )
      {
        v43 = 0;
        v19 = (_QWORD *)v4[4];
        do
        {
          v48 = *v19;
          v53 = (__int64 *)sub_1489EA0(a1, *v19);
          sub_1458920((__int64)&v54, &v53);
          v16 = v54;
          ++v19;
          v43 |= v54[(unsigned int)v55 - 1] != v48;
        }
        while ( v38 != v19 );
        if ( v43 )
        {
          v20 = sub_14813B0((_QWORD *)*a1, &v54, a3, a4);
          v16 = v54;
          v4 = (__int64 *)v20;
        }
        goto LABEL_35;
      }
      break;
    case 9:
      v54 = v56;
      v55 = 0x200000000LL;
      v14 = v4[4];
      v37 = (_QWORD *)(v14 + 8 * v4[5]);
      if ( (_QWORD *)v14 != v37 )
      {
        v42 = 0;
        v15 = (_QWORD *)v4[4];
        do
        {
          v47 = *v15;
          v53 = (__int64 *)sub_1489EA0(a1, *v15);
          sub_1458920((__int64)&v54, &v53);
          v16 = v54;
          ++v15;
          v42 |= v54[(unsigned int)v55 - 1] != v47;
        }
        while ( v37 != v15 );
        if ( v42 )
        {
          v17 = sub_147A3C0((_QWORD *)*a1, &v54, a3, a4);
          v16 = v54;
          v4 = (__int64 *)v17;
        }
LABEL_35:
        if ( v16 != v56 )
          _libc_free((unsigned __int64)v16);
      }
      break;
    case 0xA:
      if ( !sub_146CEE0(*a1, (__int64)v4, a1[5]) )
      {
        v34 = *(v4 - 1);
        if ( *(_BYTE *)(v34 + 16) == 79 )
        {
          sub_145D2A0((__int64)&v54, (__int64)a1, *(_QWORD *)(v34 - 72));
          if ( (_BYTE)v55 )
          {
            if ( sub_1455000(v54[4] + 24) )
              v36 = *(_QWORD *)(v34 - 48);
            else
              v36 = *(_QWORD *)(v34 - 24);
            v4 = (__int64 *)sub_146F1B0(*a1, v36);
          }
        }
        else
        {
          sub_145D2A0((__int64)&v54, (__int64)a1, *(v4 - 1));
          if ( (_BYTE)v55 )
            v4 = v54;
        }
      }
      break;
  }
  v53 = v4;
  sub_1466830((__int64)&v54, (__int64)(a1 + 1), &v52, (__int64 *)&v53);
  return *(__int64 **)(v56[0] + 8LL);
}
