// Function: sub_D10670
// Address: 0xd10670
//
__int64 __fastcall sub_D10670(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 result; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx

  for ( i = *(_QWORD *)(a1 + 16); *(_QWORD *)(i + 32) != a2 || *(_BYTE *)(i + 24); i += 40 )
    ;
  --*(_DWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(i + 24) )
  {
    v5 = *(_QWORD *)(i + 16);
    if ( *(_BYTE *)(v4 - 16) )
    {
      v9 = *(_QWORD *)(v4 - 24);
      if ( v9 != v5 )
      {
        if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
        {
          sub_BD60C0((_QWORD *)i);
          v9 = *(_QWORD *)(v4 - 24);
        }
        *(_QWORD *)(i + 16) = v9;
        if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
          sub_BD6050((unsigned __int64 *)i, *(_QWORD *)(v4 - 40) & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    else
    {
      *(_BYTE *)(i + 24) = 0;
      if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
        sub_BD60C0((_QWORD *)i);
    }
  }
  else if ( *(_BYTE *)(v4 - 16) )
  {
    *(_QWORD *)i = 6;
    v8 = *(_QWORD *)(v4 - 24);
    *(_QWORD *)(i + 8) = 0;
    *(_QWORD *)(i + 16) = v8;
    if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
      sub_BD6050((unsigned __int64 *)i, *(_QWORD *)(v4 - 40) & 0xFFFFFFFFFFFFFFF8LL);
    *(_BYTE *)(i + 24) = 1;
  }
  *(_QWORD *)(i + 32) = *(_QWORD *)(v4 - 8);
  result = *(_QWORD *)(a1 + 24);
  v7 = (_QWORD *)(result - 40);
  *(_QWORD *)(a1 + 24) = result - 40;
  if ( *(_BYTE *)(result - 16) )
  {
    *(_BYTE *)(result - 16) = 0;
    result = *(_QWORD *)(result - 24);
    if ( result != 0 && result != -4096 && result != -8192 )
      return sub_BD60C0(v7);
  }
  return result;
}
