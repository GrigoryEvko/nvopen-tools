// Function: sub_1B92970
// Address: 0x1b92970
//
__int64 __fastcall sub_1B92970(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 *v5; // r15
  __int64 v6; // rbx
  char v7; // si
  unsigned int v8; // r9d
  unsigned int v9; // r12d
  int i; // r12d
  __int64 v12; // rax
  int v13; // r8d
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // ecx
  int *v17; // r10
  int v18; // edi
  int v19; // ebx
  int v20; // r11d
  int v21; // r10d
  int v22; // r11d
  __int64 v23; // r10
  int v24; // r10d
  int v25; // edx
  int v26; // [rsp+4h] [rbp-7Ch]
  int v28; // [rsp+20h] [rbp-60h]
  int v29; // [rsp+28h] [rbp-58h]
  _BYTE *v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  _BYTE v32[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1;
  if ( *(_BYTE *)(a2 + 16) == 54 )
    v5 = *(__int64 **)a2;
  else
    v5 = **(__int64 ***)(a2 - 48);
  sub_1B8E090(v5, a3);
  sub_1B8DFF0(a2);
  v6 = sub_1B8F3E0(*(_QWORD *)(a1 + 384), a2);
  v28 = *(_DWORD *)v6;
  sub_16463B0(v5, *(_DWORD *)v6 * a3);
  v7 = *(_BYTE *)(a2 + 16);
  v8 = 0;
  v30 = v32;
  v31 = 0x400000000LL;
  if ( v7 == 54 && v28 )
  {
    for ( i = 0; i != v28; ++i )
    {
      v12 = *(unsigned int *)(v6 + 40);
      if ( (_DWORD)v12 )
      {
        v13 = v12 - 1;
        v14 = *(_QWORD *)(v6 + 24);
        v15 = i + *(_DWORD *)(v6 + 48);
        v16 = (v12 - 1) & (37 * v15);
        v17 = (int *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v15 == *v17 )
        {
LABEL_13:
          if ( *((_QWORD *)v17 + 1) )
          {
            if ( v8 >= HIDWORD(v31) )
            {
              sub_16CD150((__int64)&v30, v32, 0, 4, v13, v8);
              v8 = v31;
            }
            *(_DWORD *)&v30[4 * v8] = i;
            v8 = v31 + 1;
            LODWORD(v31) = v31 + 1;
          }
        }
        else
        {
          v29 = (v12 - 1) & (37 * v15);
          v20 = *v17;
          v21 = 1;
          while ( v20 != 0x7FFFFFFF )
          {
            v22 = v21 + 1;
            v23 = v13 & (unsigned int)(v29 + v21);
            v26 = v22;
            v29 = v23;
            v20 = *(_DWORD *)(v14 + 16 * v23);
            if ( v15 == v20 )
            {
              v24 = 1;
              while ( v18 != 0x7FFFFFFF )
              {
                v25 = v24 + 1;
                v16 = v13 & (v24 + v16);
                v17 = (int *)(v14 + 16LL * v16);
                v18 = *v17;
                if ( v20 == *v17 )
                  goto LABEL_13;
                v24 = v25;
              }
              v17 = (int *)(v14 + 16 * v12);
              goto LABEL_13;
            }
            v21 = v26;
          }
        }
      }
    }
    v3 = a1;
  }
  v9 = sub_14A3530(*(_QWORD *)(v3 + 328));
  if ( *(_BYTE *)(v6 + 4) )
  {
    v19 = *(_DWORD *)(v6 + 32);
    v9 += v19 * sub_14A3380(*(_QWORD *)(v3 + 328));
  }
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v9;
}
