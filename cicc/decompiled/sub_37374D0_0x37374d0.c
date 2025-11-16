// Function: sub_37374D0
// Address: 0x37374d0
//
void __fastcall sub_37374D0(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v4; // r14
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // esi
  unsigned int v9; // edx
  unsigned __int8 **v10; // rax
  unsigned __int8 *v11; // rdi
  unsigned __int64 v12; // rcx
  int v13; // eax
  int v14; // r8d

  v4 = sub_3247C80((__int64)a1, a2);
  if ( !sub_3734FE0((__int64)a1) || (unsigned __int8)sub_321F6A0(a1[26], a2) )
    v5 = a1[27] + 400;
  else
    v5 = (__int64)(a1 + 84);
  v6 = *(_QWORD *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 24);
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (unsigned __int8 **)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_5:
      v12 = (unsigned __int64)v10[1];
      if ( v12 )
      {
        if ( v4 )
          sub_32494F0(a1, (unsigned __int64)v4, 49, v12);
        return;
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != (unsigned __int8 *)-4096LL )
      {
        v14 = v13 + 1;
        v9 = v8 & (v13 + v9);
        v10 = (unsigned __int8 **)(v6 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_5;
        v13 = v14;
      }
    }
  }
  if ( v4 )
    sub_37373C0(a1, (__int64)a2, (unsigned __int64)v4);
}
