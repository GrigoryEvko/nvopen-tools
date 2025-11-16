// Function: sub_31A4C60
// Address: 0x31a4c60
//
void __fastcall sub_31A4C60(const char **a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r13
  size_t v6; // rbx
  const char *v7; // r12
  const char ***v8; // r14
  __int64 v9; // [rsp-78h] [rbp-78h]
  _QWORD v10[6]; // [rsp-68h] [rbp-68h] BYREF
  __int64 v11; // [rsp-38h] [rbp-38h] BYREF

  if ( a3 > 9 && *(_QWORD *)a2 == 0x6F6F6C2E6D766C6CLL && *(_WORD *)(a2 + 8) == 11888 && *(_BYTE *)a4 == 1 )
  {
    v4 = *(_QWORD *)(a4 + 136);
    if ( *(_BYTE *)v4 == 17 )
    {
      if ( *(_DWORD *)(v4 + 32) <= 0x40u )
        v9 = *(_QWORD *)(v4 + 24);
      else
        v9 = **(_QWORD **)(v4 + 24);
      v5 = (__int64)a1;
      v10[0] = a1;
      v6 = a3 - 10;
      v7 = *a1;
      v8 = (const char ***)v10;
      v10[1] = a1 + 2;
      v10[2] = a1 + 4;
      v10[3] = a1 + 6;
      v10[4] = a1 + 8;
      v10[5] = a1 + 10;
      if ( !v7 )
        goto LABEL_15;
LABEL_10:
      if ( strlen(v7) == v6 && (!v6 || !memcmp((const void *)(a2 + 10), v7, v6)) )
      {
LABEL_16:
        if ( sub_31A48F0(v5, v9) )
          *(_DWORD *)(v5 + 8) = v9;
      }
      else
      {
        while ( ++v8 != (const char ***)&v11 )
        {
          v5 = (__int64)*v8;
          v7 = **v8;
          if ( v7 )
            goto LABEL_10;
LABEL_15:
          if ( !v6 )
            goto LABEL_16;
        }
      }
    }
  }
}
