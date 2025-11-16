// Function: sub_D725A0
// Address: 0xd725a0
//
__int64 __fastcall sub_D725A0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // edi
  _QWORD *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r14
  unsigned __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // rdi
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  _QWORD *v22; // rax
  _BYTE *v23; // rsi
  __int64 v24; // rdx
  __int64 result; // rax
  int v26; // edx
  int v27; // r9d
  unsigned __int64 v30; // [rsp+20h] [rbp-C0h]
  __int64 *v31; // [rsp+28h] [rbp-B8h]
  __int64 *v33; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v34[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v35; // [rsp+50h] [rbp-90h]
  _BYTE *v36; // [rsp+60h] [rbp-80h] BYREF
  __int64 v37; // [rsp+68h] [rbp-78h]
  _BYTE v38[112]; // [rsp+70h] [rbp-70h] BYREF

  v6 = a4 + 8 * a5;
  v36 = v38;
  v37 = 0x400000000LL;
  v31 = &a2[a3];
  if ( v31 == a2 )
  {
    v23 = v38;
    v24 = 0;
  }
  else
  {
    v33 = a2;
    do
    {
      v7 = *v33;
      if ( v6 != a4 )
      {
        v8 = a4;
        do
        {
          v9 = *(unsigned int *)(*(_QWORD *)v8 + 24LL);
          if ( (_DWORD)v9 )
          {
            v10 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
            v11 = (v9 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
            v13 = v12[3];
            if ( v7 == v13 )
            {
LABEL_7:
              if ( v12 != (_QWORD *)(v10 + (v9 << 6)) )
              {
                v34[0] = 6;
                v14 = v12[7];
                v34[1] = 0;
                v35 = v14;
                if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
                {
                  sub_BD6050(v34, v12[5] & 0xFFFFFFFFFFFFFFF8LL);
                  v14 = v35;
                }
                if ( v14 )
                {
                  if ( v14 != -8192 && v14 != -4096 )
                    sub_BD60C0(v34);
                  v15 = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v15 == v14 + 48 )
                  {
                    v17 = 0;
                  }
                  else
                  {
                    if ( !v15 )
                      BUG();
                    v16 = *(unsigned __int8 *)(v15 - 24);
                    v17 = v15 - 24;
                    if ( (unsigned int)(v16 - 30) >= 0xB )
                      v17 = 0;
                  }
                  v18 = sub_B46EC0(v17, 0) & 0xFFFFFFFFFFFFFFFBLL;
                  v20 = (unsigned int)v37;
                  v21 = (unsigned int)v37 + 1LL;
                  if ( v21 > HIDWORD(v37) )
                  {
                    v30 = v18;
                    sub_C8D5F0((__int64)&v36, v38, v21, 0x10u, v18, v19);
                    v20 = (unsigned int)v37;
                    v18 = v30;
                  }
                  v22 = &v36[16 * v20];
                  *v22 = v14;
                  v22[1] = v18;
                  LODWORD(v37) = v37 + 1;
                }
              }
            }
            else
            {
              v26 = 1;
              while ( v13 != -4096 )
              {
                v27 = v26 + 1;
                v11 = (v9 - 1) & (v26 + v11);
                v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
                v13 = v12[3];
                if ( v7 == v13 )
                  goto LABEL_7;
                v26 = v27;
              }
            }
          }
          v8 += 8;
        }
        while ( v6 != v8 );
      }
      ++v33;
    }
    while ( v31 != v33 );
    v23 = v36;
    v24 = (unsigned int)v37;
  }
  result = sub_D724E0(a1, (__int64)v23, v24, a6);
  if ( v36 != v38 )
    return _libc_free(v36, v23);
  return result;
}
