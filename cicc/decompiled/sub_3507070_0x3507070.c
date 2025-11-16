// Function: sub_3507070
// Address: 0x3507070
//
void __fastcall sub_3507070(_QWORD *a1, __int64 *a2, signed int a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rbx
  char v12; // dl
  char v13; // al
  char v14; // cl
  int v15; // eax
  unsigned __int16 v16; // ax
  _QWORD *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned __int16 v20; // ax
  __int64 v21; // rdi
  __int64 v22; // r8
  unsigned __int64 v23; // rax
  unsigned int v24; // r9d
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  int v28; // edx
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  __int64 i; // rdi
  __int16 v32; // dx
  unsigned int v33; // edi
  __int64 v34; // r8
  unsigned int v35; // esi
  __int64 *v36; // rdx
  __int64 v37; // r9
  int v38; // edx
  int v39; // r11d
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int64 v43; // [rsp+28h] [rbp-68h]
  __int64 *v44; // [rsp+30h] [rbp-60h] BYREF
  __int64 v45; // [rsp+38h] [rbp-58h]
  _BYTE v46[80]; // [rsp+40h] [rbp-50h] BYREF

  v9 = (_QWORD *)a1[1];
  v44 = (__int64 *)v46;
  v10 = a1[2];
  v45 = 0x400000000LL;
  if ( a6 )
    sub_2E0B070(a6, (__int64)&v44, a4, a5, v9, v10);
  v43 = a4 & a5;
  v40 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v9 + 16LL) + 200LL))(*(_QWORD *)(*v9 + 16LL));
  if ( a3 < 0 )
    v11 = *(_QWORD *)(v9[7] + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v11 = *(_QWORD *)(v9[38] + 8LL * (unsigned int)a3);
  if ( v11 )
  {
    v12 = *(_BYTE *)(v11 + 4);
    if ( (v12 & 8) != 0 )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v11 + 32);
        if ( !v11 )
          break;
        v12 = *(_BYTE *)(v11 + 4);
        if ( (v12 & 8) == 0 )
          goto LABEL_7;
      }
    }
    else
    {
LABEL_7:
      v13 = *(_BYTE *)(v11 + 3);
      if ( (v13 & 0x10) == 0 )
        *(_BYTE *)(v11 + 3) = v13 & 0xBF;
      v14 = v12 & 1;
      if ( (v12 & 1) == 0 && (v12 & 2) == 0 )
      {
        v15 = *(_DWORD *)v11 >> 8;
        if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
        {
          v16 = v15 & 0xFFF;
          if ( !v16 || v43 != -1 )
            goto LABEL_16;
          v17 = (_QWORD *)(*(_QWORD *)(v40 + 272) + 16LL * v16);
          v18 = ~*v17;
          v19 = ~v17[1];
        }
        else
        {
          if ( v43 == -1 )
          {
            v20 = v15 & 0xFFF;
            if ( !v20 )
            {
              v21 = *(_QWORD *)(v11 + 16);
              v22 = *(_QWORD *)(v21 + 32);
              v23 = 0xCCCCCCCCCCCCCCCDLL * ((v11 - v22) >> 3);
              v24 = -858993459 * ((v11 - v22) >> 3);
              if ( *(_WORD *)(v21 + 68) != 68 && *(_WORD *)(v21 + 68) )
                goto LABEL_32;
              goto LABEL_27;
            }
          }
          else
          {
            v20 = v15 & 0xFFF;
            if ( !v20 )
            {
              v21 = *(_QWORD *)(v11 + 16);
              v22 = *(_QWORD *)(v21 + 32);
              v23 = 0xCCCCCCCCCCCCCCCDLL * ((v11 - v22) >> 3);
              v24 = -858993459 * ((v11 - v22) >> 3);
              if ( *(_WORD *)(v21 + 68) && *(_WORD *)(v21 + 68) != 68 )
                goto LABEL_32;
              goto LABEL_27;
            }
          }
          v27 = (__int64 *)(*(_QWORD *)(v40 + 272) + 16LL * v20);
          v18 = *v27;
          v19 = v27[1];
        }
        if ( !(a5 & v19 | a4 & v18) )
          goto LABEL_16;
        v21 = *(_QWORD *)(v11 + 16);
        v22 = *(_QWORD *)(v21 + 32);
        v23 = 0xCCCCCCCCCCCCCCCDLL * ((v11 - v22) >> 3);
        v24 = -858993459 * ((v11 - v22) >> 3);
        if ( *(_WORD *)(v21 + 68) && *(_WORD *)(v21 + 68) != 68 )
        {
          if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
          {
            v14 = (v12 & 4) != 0;
            goto LABEL_42;
          }
LABEL_32:
          v26 = v22 + 40LL * (unsigned int)v23;
          if ( !*(_BYTE *)v26 && (*(_BYTE *)(v26 + 3) & 0x10) == 0 && (*(_WORD *)(v26 + 2) & 0xFF0) != 0 )
            v14 = (*(_BYTE *)(*(_QWORD *)(v21 + 32) + 40LL * (unsigned int)sub_2E89F40(v21, v24) + 4) & 4) != 0;
LABEL_42:
          v28 = *(_DWORD *)(v21 + 44);
          v29 = v21;
          v30 = v21;
          if ( (v28 & 4) != 0 )
          {
            do
              v30 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v30 + 44) & 4) != 0 );
          }
          if ( (v28 & 8) != 0 )
          {
            do
              v29 = *(_QWORD *)(v29 + 8);
            while ( (*(_BYTE *)(v29 + 44) & 8) != 0 );
          }
          for ( i = *(_QWORD *)(v29 + 8); i != v30; v30 = *(_QWORD *)(v30 + 8) )
          {
            v32 = *(_WORD *)(v30 + 68);
            if ( (unsigned __int16)(v32 - 14) > 4u && v32 != 24 )
              break;
          }
          v33 = *(_DWORD *)(v10 + 144);
          v34 = *(_QWORD *)(v10 + 128);
          if ( v33 )
          {
            v35 = (v33 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v36 = (__int64 *)(v34 + 16LL * v35);
            v37 = *v36;
            if ( v30 == *v36 )
            {
LABEL_52:
              v25 = (v14 == 0 ? 4LL : 2LL) | v36[1] & 0xFFFFFFFFFFFFFFF8LL;
              goto LABEL_28;
            }
            v38 = 1;
            while ( v37 != -4096 )
            {
              v39 = v38 + 1;
              v35 = (v33 - 1) & (v38 + v35);
              v36 = (__int64 *)(v34 + 16LL * v35);
              v37 = *v36;
              if ( v30 == *v36 )
                goto LABEL_52;
              v38 = v39;
            }
          }
          v36 = (__int64 *)(v34 + 16LL * v33);
          goto LABEL_52;
        }
LABEL_27:
        v25 = *(_QWORD *)(*(_QWORD *)(v10 + 152)
                        + 16LL * *(unsigned int *)(*(_QWORD *)(v22 + 40LL * (v24 + 1) + 24) + 24LL)
                        + 8);
LABEL_28:
        sub_2E20270(a1, a2, v25, a3, v44, (unsigned int)v45);
      }
LABEL_16:
      while ( 1 )
      {
        v11 = *(_QWORD *)(v11 + 32);
        if ( !v11 )
          break;
        v12 = *(_BYTE *)(v11 + 4);
        if ( (v12 & 8) == 0 )
          goto LABEL_7;
      }
    }
  }
  if ( v44 != (__int64 *)v46 )
    _libc_free((unsigned __int64)v44);
}
