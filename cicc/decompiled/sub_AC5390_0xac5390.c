// Function: sub_AC5390
// Address: 0xac5390
//
__int64 __fastcall sub_AC5390(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  v3 = (unsigned int *)sub_AC5300(a2, a3);
  v4 = *(_DWORD *)(sub_AC5230(a2) + 8) >> 8;
  if ( v4 == 32 )
  {
    v9 = *v3;
    *(_DWORD *)(a1 + 8) = 32;
    *(_QWORD *)a1 = v9;
    return a1;
  }
  else
  {
    if ( v4 <= 0x20 )
    {
      if ( v4 == 8 )
      {
        v5 = *(unsigned __int8 *)v3;
        *(_DWORD *)(a1 + 8) = 8;
        *(_QWORD *)a1 = v5;
        return a1;
      }
      if ( v4 == 16 )
      {
        v7 = *(unsigned __int16 *)v3;
        *(_DWORD *)(a1 + 8) = 16;
        *(_QWORD *)a1 = v7;
        return a1;
      }
LABEL_10:
      BUG();
    }
    if ( v4 != 64 )
      goto LABEL_10;
    v8 = *(_QWORD *)v3;
    *(_DWORD *)(a1 + 8) = 64;
    *(_QWORD *)a1 = v8;
    return a1;
  }
}
