// Function: sub_16E7EE0
// Address: 0x16e7ee0
//
__int64 __fastcall sub_16E7EE0(__int64 a1, char *a2, size_t a3)
{
  size_t v3; // r13
  __int64 v4; // rax
  size_t v5; // rbx
  size_t v7; // r14
  __int64 v8; // rdx
  size_t v9; // rdx
  size_t v10; // r13
  size_t v11; // r15

  v3 = a3;
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 16) - v4;
  if ( a3 > v5 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      if ( !v8 )
        break;
      if ( v4 != v8 )
      {
        v3 -= v5;
        sub_16E7E70(a1, (unsigned __int8 *)a2, v5);
        sub_16E7BA0((__int64 *)a1);
        a2 += v5;
LABEL_8:
        v4 = *(_QWORD *)(a1 + 24);
        v5 = *(_QWORD *)(a1 + 16) - v4;
        if ( v5 >= v3 )
          goto LABEL_2;
        v7 = v3;
        goto LABEL_4;
      }
      v9 = v3 % v5;
      v10 = v3 - v3 % v5;
      v11 = v9;
      v7 = v9;
      (*(void (__fastcall **)(__int64, char *, size_t))(*(_QWORD *)a1 + 56LL))(a1, a2, v10);
      v4 = *(_QWORD *)(a1 + 24);
      v5 = *(_QWORD *)(a1 + 16) - v4;
      a2 += v10;
      if ( v11 <= v5 )
      {
        sub_16E7E70(a1, (unsigned __int8 *)a2, v11);
        return a1;
      }
LABEL_4:
      v3 = v7;
    }
    if ( !*(_DWORD *)(a1 + 32) )
    {
      (*(void (__fastcall **)(__int64, char *, size_t))(*(_QWORD *)a1 + 56LL))(a1, a2, v3);
      return a1;
    }
    sub_16E7D60((__int64 *)a1);
    goto LABEL_8;
  }
LABEL_2:
  sub_16E7E70(a1, (unsigned __int8 *)a2, v3);
  return a1;
}
