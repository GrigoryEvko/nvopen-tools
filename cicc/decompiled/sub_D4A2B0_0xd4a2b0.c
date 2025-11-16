// Function: sub_D4A2B0
// Address: 0xd4a2b0
//
__int64 __fastcall sub_D4A2B0(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // rax
  int *v9; // rdx
  unsigned int v10; // eax
  int v11; // ecx
  __int64 v13; // [rsp+0h] [rbp-10h]

  v6 = sub_D4A110(a1, a2, a3, a4, a5, a6);
  v13 = v6;
  if ( v7 && v6 && *(_QWORD *)v6 && (v8 = *(_QWORD *)(*(_QWORD *)v6 + 136LL)) != 0 )
  {
    v9 = *(int **)(v8 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
    {
      v11 = *v9;
    }
    else
    {
      v11 = 0;
      if ( v10 )
        v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
    }
    LODWORD(v13) = v11;
    BYTE4(v13) = 1;
    return v13;
  }
  else
  {
    BYTE4(v13) = 0;
    return v13;
  }
}
