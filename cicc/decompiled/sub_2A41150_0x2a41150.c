// Function: sub_2A41150
// Address: 0x2a41150
//
void __fastcall sub_2A41150(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // [rsp-30h] [rbp-30h] BYREF

  if ( a1 && !sub_B2FC80(a1) )
  {
    v6 = *(__int64 **)(a1 - 32);
    v7 = 32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v6 + 7) & 0x40) != 0 )
    {
      v8 = (__int64 *)*(v6 - 1);
      v6 = &v8[(unsigned __int64)v7 / 8];
    }
    else
    {
      v8 = &v6[v7 / 0xFFFFFFFFFFFFFFF8LL];
    }
    while ( v6 != v8 )
    {
      v9 = *v8;
      v8 += 4;
      v10 = v9;
      sub_2A40B10(a2, &v10, v2, v3, v4, v5);
    }
  }
}
