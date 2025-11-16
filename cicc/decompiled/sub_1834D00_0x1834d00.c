// Function: sub_1834D00
// Address: 0x1834d00
//
unsigned __int64 *__fastcall sub_1834D00(_QWORD *a1, _QWORD *a2, unsigned __int64 **a3)
{
  unsigned __int64 *v5; // r12
  unsigned __int64 *v6; // rax
  _QWORD *v7; // r14
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // r8
  _BOOL8 v13; // rdi
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  signed __int64 v18; // rdi
  _QWORD *v19; // [rsp+8h] [rbp-38h]
  _QWORD *v20; // [rsp+8h] [rbp-38h]

  v5 = (unsigned __int64 *)sub_22077B0(72);
  v6 = *a3;
  v5[4] = **a3;
  v7 = (_QWORD *)v6[1];
  v8 = v6[2];
  v9 = v6[3];
  v6[2] = 0;
  v6[3] = 0;
  v6[1] = 0;
  v5[5] = (unsigned __int64)v7;
  v5[6] = v8;
  v5[7] = v9;
  v5[8] = 0;
  v10 = sub_18349D0(a1, a2, v5 + 4);
  if ( v11 )
  {
    v12 = a1 + 1;
    v13 = 1;
    if ( !v10 && v11 != v12 )
    {
      v15 = v11[4];
      if ( v5[4] >= v15 )
      {
        v13 = 0;
        if ( v5[4] == v15 )
        {
          v16 = (_QWORD *)v11[5];
          v17 = (_QWORD *)v8;
          v18 = v11[6] - (_QWORD)v16;
          if ( (__int64)(v8 - (_QWORD)v7) > v18 )
            v17 = (_QWORD *)((char *)v7 + v18);
          if ( v7 == v17 )
          {
LABEL_18:
            v13 = v11[6] != (_QWORD)v16;
          }
          else
          {
            while ( 1 )
            {
              if ( *v7 < *v16 )
              {
                v13 = 1;
                goto LABEL_3;
              }
              if ( *v7 > *v16 )
                break;
              ++v7;
              ++v16;
              if ( v17 == v7 )
                goto LABEL_18;
            }
            v13 = 0;
          }
        }
      }
    }
LABEL_3:
    sub_220F040(v13, v5, v11, a1 + 1);
    ++a1[5];
    return v5;
  }
  else
  {
    if ( v7 )
    {
      v19 = v10;
      j_j___libc_free_0(v7, v9 - (_QWORD)v7);
      v10 = v19;
    }
    v20 = v10;
    j_j___libc_free_0(v5, 72);
    return v20;
  }
}
