// Function: sub_6794F0
// Address: 0x6794f0
//
__int64 __fastcall sub_6794F0(__int64 **a1, __int64 a2, int a3)
{
  unsigned int *v3; // r13
  __int64 *v6; // rax
  __int64 *v7; // rbx
  int v8; // eax
  char v9; // al
  _BOOL8 v10; // rdx
  __int64 v11; // rdi
  unsigned int *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx

  v3 = (unsigned int *)a2;
  v6 = a1[5];
  if ( v6 && *((_BYTE *)v6 + 24) )
    sub_6851C0(307, &dword_4F063F8);
  if ( a3 )
  {
    v7 = *a1;
    if ( *a1 )
    {
      v8 = 0;
      do
      {
        if ( (v7[4] & 4) == 0 )
        {
          if ( !v8 )
            sub_6851C0(306, &dword_4F063F8);
          v9 = *((_BYTE *)v7 + 32) | 4;
          *((_BYTE *)v7 + 32) = v9;
          *((_BYTE *)v7 + 32) = (_BYTE)a1[4] & 8 | v9 & 0xF7;
          v7[5] = sub_7305B0();
          v8 = 1;
        }
        v7 = (__int64 *)*v7;
      }
      while ( v7 );
    }
  }
  v10 = 0;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 10) <= 1u )
    v10 = (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 193LL) & 4) != 0;
  sub_6D05D0(a1, 1, v10);
  v11 = (__int64)a1[7];
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 10) >= 2u )
    v3 = 0;
  v12 = v3;
  sub_85E7F0(v11, v3, ((_BYTE)a1[4] & 8) != 0);
  if ( word_4F06418[0] != 9 )
  {
    v12 = &dword_4F063F8;
    v11 = 807;
    sub_6851C0(807, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(807, &dword_4F063F8, v13, v14);
  }
  return sub_7B8B50(v11, v12, v13, v14);
}
