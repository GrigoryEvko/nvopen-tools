// Function: sub_844910
// Address: 0x844910
//
void __fastcall sub_844910(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rax
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax

  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    v6 = sub_844780(*(__int64 **)(a1 + 144), 1, a3, a4, a5);
    if ( v6 )
    {
      sub_82B7B0(v6, a2);
      if ( a2 )
        sub_6F4D20((__m128i *)a1, 0, 0, v7, v8, v9);
    }
    else
    {
      v10 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624);
      if ( v10 )
      {
        v11 = *v10;
        if ( v11 )
        {
          v12 = *(_BYTE *)(v11 + 80);
          if ( v12 == 9 || v12 == 7 )
          {
            v13 = *(_QWORD *)(v11 + 88);
          }
          else
          {
            if ( v12 != 21 )
              return;
            v13 = *(_QWORD *)(*(_QWORD *)(v11 + 88) + 192LL);
          }
          if ( v13 )
            *(_BYTE *)(v13 + 176) |= 0x10u;
        }
      }
    }
  }
}
