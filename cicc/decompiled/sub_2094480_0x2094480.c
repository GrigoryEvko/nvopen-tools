// Function: sub_2094480
// Address: 0x2094480
//
__int64 __fastcall sub_2094480(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 i; // rbx
  __int64 v5; // rax
  _BYTE *v6; // rax
  __int64 v7; // rax
  char v8; // dl
  _WORD *v9; // rdx
  char v10; // [rsp-68h] [rbp-68h] BYREF
  __int64 v11; // [rsp-60h] [rbp-60h]
  char *v12[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v13[9]; // [rsp-48h] [rbp-48h] BYREF

  result = *(unsigned int *)(a1 + 60);
  if ( (_DWORD)result )
  {
    v3 = (unsigned int)(result - 1);
    for ( i = 0; ; ++i )
    {
      v7 = *(_QWORD *)(a1 + 40) + 16 * i;
      v8 = *(_BYTE *)v7;
      if ( *(_BYTE *)v7 != 1 )
        break;
      v9 = *(_WORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 1u )
      {
        result = sub_16E7EE0(a2, "ch", 2u);
        goto LABEL_5;
      }
      result = 26723;
      *v9 = 26723;
      *(_QWORD *)(a2 + 24) += 2LL;
      if ( i == v3 )
        return result;
LABEL_6:
      if ( (_DWORD)i != -1 )
      {
        v6 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v6 )
        {
          sub_16E7EE0(a2, ",", 1u);
        }
        else
        {
          *v6 = 44;
          ++*(_QWORD *)(a2 + 24);
        }
      }
    }
    v5 = *(_QWORD *)(v7 + 8);
    v10 = v8;
    v11 = v5;
    sub_1F596C0((__int64)v12, &v10);
    sub_16E7EE0(a2, v12[0], (size_t)v12[1]);
    result = (__int64)v13;
    if ( (_QWORD *)v12[0] != v13 )
      result = j_j___libc_free_0(v12[0], v13[0] + 1LL);
LABEL_5:
    if ( i == v3 )
      return result;
    goto LABEL_6;
  }
  return result;
}
