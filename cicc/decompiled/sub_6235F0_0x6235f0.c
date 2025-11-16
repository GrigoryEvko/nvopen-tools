// Function: sub_6235F0
// Address: 0x6235f0
//
void __fastcall sub_6235F0(_QWORD *a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx

  if ( *a1 )
  {
    if ( (unsigned int)sub_731A30() )
    {
      v5 = sub_867B10();
      v8 = *a1;
      if ( (*(_BYTE *)(*a1 - 8LL) & 1) != 0 )
      {
        sub_7296B0(*(unsigned int *)(*(_QWORD *)(v5 + 32) + 164LL), a2, v6, v7);
        *a1 = sub_73B8B0(*a1, 0x2000);
        sub_7296B0(unk_4F073B8, 0x2000, v9, v10);
        v8 = *a1;
      }
      sub_72D910(v8, (unsigned int)(a3 + 4), a2);
      *a1 = 0;
    }
    else if ( (*(_BYTE *)(*a1 - 8LL) & 1) == 0 )
    {
      *a1 = sub_73B8B0(*a1, 0x10000);
    }
  }
}
