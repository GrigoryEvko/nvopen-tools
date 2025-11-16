// Function: sub_F07530
// Address: 0xf07530
//
void __fastcall sub_F07530(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  const void *v3; // rax

  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( *(_DWORD *)(a1 + 8) <= 0x40u && *(_DWORD *)(a2 + 8) <= 0x40u )
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    }
    else
    {
      sub_C43990(a1, a2);
    }
  }
  else
  {
    v2 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v2;
    if ( v2 > 0x40 )
    {
      sub_C43780(a1, (const void **)a2);
      *(_BYTE *)(a1 + 16) = 1;
    }
    else
    {
      v3 = *(const void **)a2;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v3;
    }
  }
}
