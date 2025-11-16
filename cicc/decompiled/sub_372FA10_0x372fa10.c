// Function: sub_372FA10
// Address: 0x372fa10
//
__int64 __fastcall sub_372FA10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v3; // r14
  __int64 v4; // rcx
  __int64 v5; // r12
  __int64 v6; // r15
  _BYTE *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // rdx
  _BYTE *v10; // rax
  __int64 *v11; // rdi
  size_t v12; // rdx
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // rbx
  __int64 v16; // [rsp+0h] [rbp-40h]

  if ( a3 == a2 )
    return a2;
  v3 = *(unsigned __int64 **)(a1 + 8);
  v4 = a3;
  v5 = (__int64)v3 - a3;
  if ( (unsigned __int64 *)a3 == v3 )
    goto LABEL_15;
  v6 = v5 >> 5;
  if ( v5 <= 0 )
    goto LABEL_15;
  v7 = (_BYTE *)(a3 + 16);
  v8 = (__int64 *)(a2 + 16);
  do
  {
    v10 = (_BYTE *)*((_QWORD *)v7 - 2);
    v11 = (__int64 *)*(v8 - 2);
    if ( v10 == v7 )
    {
      v12 = *((_QWORD *)v7 - 1);
      if ( v12 )
      {
        if ( v12 == 1 )
        {
          *(_BYTE *)v11 = *v7;
          v12 = *((_QWORD *)v7 - 1);
          v11 = (__int64 *)*(v8 - 2);
        }
        else
        {
          v16 = v4;
          memcpy(v11, v7, v12);
          v12 = *((_QWORD *)v7 - 1);
          v11 = (__int64 *)*(v8 - 2);
          v4 = v16;
        }
      }
      *(v8 - 1) = v12;
      *((_BYTE *)v11 + v12) = 0;
      v11 = (__int64 *)*((_QWORD *)v7 - 2);
    }
    else
    {
      if ( v11 == v8 )
      {
        *(v8 - 2) = (__int64)v10;
        *(v8 - 1) = *((_QWORD *)v7 - 1);
        *v8 = *(_QWORD *)v7;
      }
      else
      {
        *(v8 - 2) = (__int64)v10;
        v9 = *v8;
        *(v8 - 1) = *((_QWORD *)v7 - 1);
        *v8 = *(_QWORD *)v7;
        if ( v11 )
        {
          *((_QWORD *)v7 - 2) = v11;
          *(_QWORD *)v7 = v9;
          goto LABEL_8;
        }
      }
      *((_QWORD *)v7 - 2) = v7;
      v11 = (__int64 *)v7;
    }
LABEL_8:
    *((_QWORD *)v7 - 1) = 0;
    v8 += 4;
    v7 += 32;
    *(_BYTE *)v11 = 0;
    --v6;
  }
  while ( v6 );
  v3 = *(unsigned __int64 **)(a1 + 8);
  v5 = (__int64)v3 - v4;
LABEL_15:
  v13 = (unsigned __int64 *)(a2 + v5);
  if ( v13 != v3 )
  {
    v14 = v13;
    do
    {
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14);
      v14 += 4;
    }
    while ( v14 != v3 );
    *(_QWORD *)(a1 + 8) = v13;
  }
  return a2;
}
