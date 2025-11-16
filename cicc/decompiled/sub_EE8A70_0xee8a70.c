// Function: sub_EE8A70
// Address: 0xee8a70
//
__int64 __fastcall sub_EE8A70(__int64 a1)
{
  char *v1; // rax
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rdx
  char *v5; // rdx
  __int64 v6; // rdi
  char v7; // r8
  char *v8; // rax
  char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-20h] BYREF
  __int64 v12; // [rsp+8h] [rbp-18h]

  v1 = *(char **)a1;
  v2 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
  if ( v2 <= 2 )
  {
    if ( v2 != 2 )
      goto LABEL_10;
  }
  else if ( *(_WORD *)v1 == 28774 && v1[2] == 84 )
  {
    v6 = a1 + 808;
    *(_QWORD *)(v6 - 808) = v1 + 3;
    return sub_EE68C0(v6, "this");
  }
  if ( *(_WORD *)v1 == 28774 )
  {
    *(_QWORD *)a1 = v1 + 2;
    sub_EE3340(a1);
    v11 = sub_EE32C0((char **)a1, 0);
    result = 0;
    v12 = v4;
    v5 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v5 == 95 )
    {
      *(_QWORD *)a1 = v5 + 1;
      return sub_EE88B0(a1 + 808, &v11);
    }
    return result;
  }
LABEL_10:
  v7 = sub_EE3B50((const void **)a1, 2u, "fL");
  result = 0;
  if ( v7 )
  {
    if ( sub_EE32C0((char **)a1, 0)
      && (v8 = *(char **)a1, *(_QWORD *)a1 != *(_QWORD *)(a1 + 8))
      && *v8 == 112
      && (*(_QWORD *)a1 = v8 + 1,
          sub_EE3340(a1),
          v11 = sub_EE32C0((char **)a1, 0),
          v9 = *(char **)a1,
          v12 = v10,
          v9 != *(char **)(a1 + 8))
      && *v9 == 95 )
    {
      *(_QWORD *)a1 = v9 + 1;
      return sub_EE88B0(a1 + 808, &v11);
    }
    else
    {
      return 0;
    }
  }
  return result;
}
