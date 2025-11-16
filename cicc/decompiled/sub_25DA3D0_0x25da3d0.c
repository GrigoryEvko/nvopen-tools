// Function: sub_25DA3D0
// Address: 0x25da3d0
//
void __fastcall sub_25DA3D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned int v5; // ebx

  if ( (_BYTE)qword_4FF1588 )
  {
    v3 = sub_BA91D0(a2, "Virtual Function Elim", 0x15u);
    if ( v3 )
    {
      if ( *(_BYTE *)v3 == 1 )
      {
        v4 = *(_QWORD *)(v3 + 136);
        if ( *(_BYTE *)v4 == 17 )
        {
          v5 = *(_DWORD *)(v4 + 32);
          if ( v5 <= 0x40 )
          {
            if ( !*(_QWORD *)(v4 + 24) )
              return;
          }
          else if ( v5 == (unsigned int)sub_C444A0(v4 + 24) )
          {
            return;
          }
          sub_25D9D90(a1, a2);
          if ( *(_DWORD *)(a1 + 492) != *(_DWORD *)(a1 + 496) )
            sub_25D9AE0(a1, a2);
        }
      }
    }
  }
}
