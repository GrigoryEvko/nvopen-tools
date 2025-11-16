// Function: sub_E22720
// Address: 0xe22720
//
__int64 __fastcall sub_E22720(__int64 a1, __int64 *a2, int a3)
{
  __int64 result; // rax
  char *v5; // rcx
  char v6; // di
  char v7; // si
  int v8; // edx
  char v9; // al
  _QWORD *v10; // rdx
  char v11; // r13
  __int64 *v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rdx

  result = *a2;
  if ( *a2 )
  {
    v5 = (char *)a2[1];
    v6 = *v5;
    *a2 = result - 1;
    a2[1] = (__int64)(v5 + 1);
    if ( a3 == 1 )
    {
      v7 = v6;
      v8 = 1;
    }
    else if ( a3 == 2 )
    {
      if ( v6 == 75 )
        return sub_E22610(a1, a2);
      v7 = v6;
      v8 = 2;
    }
    else
    {
      if ( v6 <= 49 )
      {
        if ( v6 > 47 )
          return sub_E214D0(a1, (__int64)a2, v6 == 49);
      }
      else if ( v6 == 66 )
      {
        return sub_E215C0(a1);
      }
      v7 = v6;
      v8 = 0;
    }
    v9 = sub_E216A0(a1, v7, v8);
    v10 = *(_QWORD **)(a1 + 16);
    v11 = v9;
    result = (*v10 + v10[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v10[1] = result - *v10 + 32;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v12 = (__int64 *)sub_22077B0(32);
      v13 = v12;
      if ( v12 )
      {
        *v12 = 0;
        v12[1] = 0;
        v12[2] = 0;
        v12[3] = 0;
      }
      result = sub_2207820(4096);
      *v13 = result;
      v14 = *(_QWORD *)(a1 + 16);
      v13[2] = 4096;
      v13[3] = v14;
      *(_QWORD *)(a1 + 16) = v13;
      v13[1] = 32;
      if ( result )
        goto LABEL_14;
    }
    else
    {
      if ( result )
      {
LABEL_14:
        *(_DWORD *)(result + 8) = 8;
        *(_QWORD *)(result + 16) = 0;
        *(_BYTE *)(result + 24) = v11;
        *(_QWORD *)result = &unk_49E0FB0;
        return result;
      }
      return 0;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
  }
  return result;
}
