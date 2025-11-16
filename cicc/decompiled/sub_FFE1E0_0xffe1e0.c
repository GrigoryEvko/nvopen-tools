// Function: sub_FFE1E0
// Address: 0xffe1e0
//
__int64 __fastcall sub_FFE1E0(_QWORD *a1, _BYTE *a2, __int64 *a3, _DWORD *a4)
{
  __int64 v6; // r12
  __int64 v8; // r9
  _BYTE *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  int v16; // eax
  int v17; // [rsp+4h] [rbp-8Ch]
  int v18; // [rsp+8h] [rbp-88h]
  int v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+8h] [rbp-88h]
  int v21; // [rsp+8h] [rbp-88h]
  _BYTE *v22; // [rsp+10h] [rbp-80h] BYREF
  __int64 v23; // [rsp+18h] [rbp-78h]
  _BYTE src[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = (__int64)a2;
  sub_ED2710((__int64)&v22, (__int64)a2, 0, qword_4F8EB68, a3, 0);
  v9 = v22;
  if ( v22 == src )
  {
    v10 = (unsigned int)v23;
    v11 = *((unsigned int *)a1 + 2);
    v12 = (unsigned int)v23;
    if ( (unsigned int)v23 <= v11 )
    {
      v13 = src;
      if ( (_DWORD)v23 )
      {
        v19 = v23;
        a2 = src;
        memmove((void *)*a1, src, 16LL * (unsigned int)v23);
        v13 = v22;
        LODWORD(v12) = v19;
      }
    }
    else
    {
      if ( (unsigned int)v23 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        *((_DWORD *)a1 + 2) = 0;
        v21 = v10;
        sub_C8D5F0((__int64)a1, a1 + 2, v10, 0x10u, v12, v8);
        v13 = v22;
        v10 = (unsigned int)v23;
        v11 = 0;
        LODWORD(v12) = v21;
        a2 = v22;
      }
      else
      {
        a2 = src;
        v13 = src;
        if ( *((_DWORD *)a1 + 2) )
        {
          v17 = v23;
          v20 = 16 * v11;
          memmove((void *)*a1, src, 16 * v11);
          v13 = v22;
          v10 = (unsigned int)v23;
          LODWORD(v12) = v17;
          a2 = &v22[v20];
          v11 = v20;
        }
      }
      v14 = 16 * v10;
      if ( a2 != &v13[v14] )
      {
        v18 = v12;
        memcpy((void *)(v11 + *a1), a2, v14 - v11);
        v13 = v22;
        LODWORD(v12) = v18;
      }
    }
    *((_DWORD *)a1 + 2) = v12;
    if ( v13 != src )
      _libc_free(v13, a2);
    if ( !*((_DWORD *)a1 + 2) )
    {
LABEL_11:
      *a4 = 0;
      return 0;
    }
  }
  else
  {
    if ( (_QWORD *)*a1 != a1 + 2 )
    {
      _libc_free(*a1, a2);
      v9 = v22;
    }
    *a1 = v9;
    v16 = v23;
    a1[1] = v23;
    if ( !v16 )
      goto LABEL_11;
  }
  *a4 = sub_FFE160((__int64)a1, v6, *a3);
  return *a1;
}
