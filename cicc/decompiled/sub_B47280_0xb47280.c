// Function: sub_B47280
// Address: 0xb47280
//
__int64 __fastcall sub_B47280(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // r14
  unsigned __int8 v5; // dl
  bool v6; // si
  __int64 v7; // r15
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r10
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned __int8 v19; // dl
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+8h] [rbp-68h]
  _BYTE *v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+18h] [rbp-58h]
  _BYTE v30[80]; // [rsp+20h] [rbp-50h] BYREF

  result = sub_BC89C0();
  if ( result )
  {
    v3 = result;
    v4 = (unsigned int)sub_BC8810(result);
    result = (*(_BYTE *)(v3 - 16) & 2) != 0 ? *(unsigned int *)(v3 - 24) : (*(_WORD *)(v3 - 16) >> 6) & 0xFu;
    if ( (_DWORD)v4 + 2 == (_DWORD)result )
    {
      v28 = v30;
      v29 = 0x400000000LL;
      v5 = *(_BYTE *)(v3 - 16);
      v6 = (v5 & 2) != 0;
      result = (v5 & 2) != 0 ? *(unsigned int *)(v3 - 24) : (*(_WORD *)(v3 - 16) >> 6) & 0xFu;
      if ( (int)v4 + 2 >= (unsigned int)result )
      {
        if ( (_DWORD)v4 )
        {
          v7 = 0;
          v8 = 4;
          v9 = 0;
          v10 = 8LL * (unsigned int)(v4 - 1);
          while ( 1 )
          {
            if ( v6 )
            {
              v11 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + v7);
              v12 = v9 + 1;
              if ( v9 + 1 <= v8 )
                goto LABEL_11;
            }
            else
            {
              v11 = *(_QWORD *)(v3 + -16 - 8LL * ((v5 >> 2) & 0xF) + v7);
              v12 = v9 + 1;
              if ( v9 + 1 <= v8 )
                goto LABEL_11;
            }
            v25 = v10;
            v26 = v11;
            sub_C8D5F0(&v28, v30, v12, 8);
            v9 = (unsigned int)v29;
            v10 = v25;
            v11 = v26;
LABEL_11:
            *(_QWORD *)&v28[8 * v9] = v11;
            v9 = (unsigned int)(v29 + 1);
            LODWORD(v29) = v29 + 1;
            v5 = *(_BYTE *)(v3 - 16);
            v6 = (v5 & 2) != 0;
            if ( v10 == v7 )
            {
              v13 = HIDWORD(v29);
              v14 = v9 + 1;
              goto LABEL_17;
            }
            v8 = HIDWORD(v29);
            v7 += 8;
          }
        }
        v13 = 4;
        v14 = 1;
        v9 = 0;
LABEL_17:
        v15 = v3 - 16;
        if ( v6 )
          v16 = *(_QWORD *)(v3 - 32);
        else
          v16 = v15 - 8LL * ((v5 >> 2) & 0xF);
        v17 = *(_QWORD *)(v16 + 8LL * (unsigned int)(v4 + 1));
        if ( v13 < v14 )
        {
          v27 = *(_QWORD *)(v16 + 8LL * (unsigned int)(v4 + 1));
          sub_C8D5F0(&v28, v30, v14, 8);
          v9 = (unsigned int)v29;
          v17 = v27;
        }
        *(_QWORD *)&v28[8 * v9] = v17;
        v18 = (unsigned int)(v29 + 1);
        LODWORD(v29) = v29 + 1;
        v19 = *(_BYTE *)(v3 - 16);
        if ( (v19 & 2) != 0 )
          v20 = *(_QWORD *)(v3 - 32);
        else
          v20 = v15 - 8LL * ((v19 >> 2) & 0xF);
        v21 = *(_QWORD *)(v20 + 8 * v4);
        if ( v18 + 1 > (unsigned __int64)HIDWORD(v29) )
        {
          sub_C8D5F0(&v28, v30, v18 + 1, 8);
          v18 = (unsigned int)v29;
        }
        *(_QWORD *)&v28[8 * v18] = v21;
        v22 = (unsigned int)(v29 + 1);
        LODWORD(v29) = v29 + 1;
        v23 = (_QWORD *)(*(_QWORD *)(v3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v3 + 8) & 4) != 0 )
          v23 = (_QWORD *)*v23;
        v24 = sub_B9C770(v23, v28, v22, 0, 1);
        result = sub_B99FD0(a1, 2, v24);
        if ( v28 != v30 )
          return _libc_free(v28, 2);
      }
    }
  }
  return result;
}
