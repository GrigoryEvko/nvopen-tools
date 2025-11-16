// Function: sub_B19F20
// Address: 0xb19f20
//
__int64 __fastcall sub_B19F20(__int64 a1, char *a2, __int64 a3)
{
  unsigned __int8 v3; // bl
  char *v4; // r15
  __int64 v6; // r8
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-58h]
  char v12; // [rsp+17h] [rbp-49h]
  __int64 v13; // [rsp+18h] [rbp-48h]
  __int64 v14[8]; // [rsp+20h] [rbp-40h] BYREF

  v3 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 1;
  v4 = *(char **)(a3 + 24);
  if ( *v4 != 84 )
  {
    v6 = *((_QWORD *)v4 + 5);
    if ( v6 )
      goto LABEL_4;
LABEL_13:
    v7 = 0;
    v8 = 0;
    goto LABEL_5;
  }
  v6 = *(_QWORD *)(*((_QWORD *)v4 - 1)
                 + 32LL * *((unsigned int *)v4 + 18)
                 + 8LL * (unsigned int)((a3 - *((_QWORD *)v4 - 1)) >> 5));
  if ( !v6 )
    goto LABEL_13;
LABEL_4:
  v7 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
  v8 = *(_DWORD *)(v6 + 44) + 1;
LABEL_5:
  v12 = *v4;
  v13 = v6;
  if ( v8 >= *(_DWORD *)(a1 + 32) || !*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v7) )
    return 1;
  v11 = *((_QWORD *)a2 + 5);
  result = sub_B192B0(a1, v11);
  if ( (_BYTE)result )
  {
    if ( v3 == 34 )
    {
      v10 = *((_QWORD *)a2 - 12);
      v14[0] = v11;
      v14[1] = v10;
      return sub_B19ED0(a1, v14, a3);
    }
    else if ( v13 == v11 )
    {
      if ( v12 != 84 )
        return sub_B445A0(a2, v4);
    }
    else
    {
      return sub_B19720(a1, v11, v13);
    }
  }
  return result;
}
