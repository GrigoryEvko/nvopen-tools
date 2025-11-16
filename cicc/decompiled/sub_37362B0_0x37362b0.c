// Function: sub_37362B0
// Address: 0x37362b0
//
__int64 __fastcall sub_37362B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // rdi
  int v10; // edx
  int v11; // r9d

  if ( !sub_3734FE0(a1) || (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a2) )
  {
    v3 = *(_QWORD *)(a1 + 216);
    v4 = *(unsigned int *)(v3 + 456);
    v5 = *(_QWORD *)(v3 + 440);
    if ( !(_DWORD)v4 )
      return 0;
  }
  else
  {
    v4 = *(unsigned int *)(a1 + 728);
    v5 = *(_QWORD *)(a1 + 712);
    if ( !(_DWORD)v4 )
      return 0;
  }
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
  {
LABEL_4:
    if ( v7 != (__int64 *)(v5 + 16 * v4) )
      return v7[1];
  }
  else
  {
    v10 = 1;
    while ( v8 != -4096 )
    {
      v11 = v10 + 1;
      v6 = (v4 - 1) & (v10 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_4;
      v10 = v11;
    }
  }
  return 0;
}
