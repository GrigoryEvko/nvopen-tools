// Function: sub_31C1020
// Address: 0x31c1020
//
void __fastcall sub_31C1020(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rdx
  char *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 *v14; // rdi
  __int64 *v15; // rbx
  __int64 *v16; // r13
  _BYTE *v17; // rsi
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // [rsp+18h] [rbp-88h] BYREF
  __int64 *v21; // [rsp+20h] [rbp-80h] BYREF
  __int64 v22; // [rsp+28h] [rbp-78h]
  _BYTE v23[112]; // [rsp+30h] [rbp-70h] BYREF

  v7 = (char *)a1[2];
  v8 = (char *)a1[1];
  v22 = 0x800000000LL;
  v21 = (__int64 *)v23;
  if ( (unsigned __int64)(v7 - v8) > 0x40 )
  {
    sub_C8D5F0((__int64)&v21, v23, (v7 - v8) >> 3, 8u, a5, a6);
    v7 = (char *)a1[2];
    v8 = (char *)a1[1];
    if ( v7 == v8 )
    {
      v10 = (unsigned int)v22;
    }
    else
    {
      do
      {
LABEL_8:
        v12 = *(_QWORD *)v8;
        if ( v7 - v8 <= 8 )
        {
          v9 = (unsigned int)v22;
          a1[2] -= 8;
          v10 = v9;
          if ( v12 == a2 )
            goto LABEL_10;
        }
        else
        {
          v13 = *((_QWORD *)v7 - 1);
          *((_QWORD *)v7 - 1) = v12;
          sub_31C0E70((__int64)v8, 0, (v7 - 8 - v8) >> 3, v13);
          v9 = (unsigned int)v22;
          a1[2] -= 8;
          v10 = v9;
          if ( v12 == a2 )
            goto LABEL_10;
        }
        if ( v9 + 1 > (unsigned __int64)HIDWORD(v22) )
        {
          sub_C8D5F0((__int64)&v21, v23, v9 + 1, 8u, a5, a6);
          v9 = (unsigned int)v22;
        }
        v21[v9] = v12;
        v7 = (char *)a1[2];
        v8 = (char *)a1[1];
        v11 = v22 + 1;
        LODWORD(v22) = v22 + 1;
      }
      while ( v7 != v8 );
      v10 = v11;
    }
LABEL_10:
    v14 = v21;
    v15 = &v21[v10];
    if ( v15 != v21 )
    {
      v16 = v21;
      do
      {
        v19 = *v16;
        v17 = (_BYTE *)a1[2];
        v20 = *v16;
        if ( v17 == (_BYTE *)a1[3] )
        {
          sub_31C0410((__int64)(a1 + 1), v17, &v20);
          v18 = (_BYTE *)a1[2];
        }
        else
        {
          if ( v17 )
          {
            *(_QWORD *)v17 = v19;
            v17 = (_BYTE *)a1[2];
          }
          v18 = v17 + 8;
          a1[2] = (__int64)v18;
        }
        ++v16;
        sub_31BFEC0(a1[1], ((__int64)&v18[-a1[1]] >> 3) - 1, 0, *((_QWORD *)v18 - 1));
      }
      while ( v15 != v16 );
      v14 = v21;
    }
    if ( v14 != (__int64 *)v23 )
      _libc_free((unsigned __int64)v14);
  }
  else if ( v8 != v7 )
  {
    goto LABEL_8;
  }
}
