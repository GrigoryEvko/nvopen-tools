// Function: sub_1595AB0
// Address: 0x1595ab0
//
__int64 __fastcall sub_1595AB0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  v3 = (unsigned int *)sub_1595950(a2, a3);
  v4 = *(_DWORD *)(sub_1595890(a2) + 8) >> 8;
  if ( v4 == 32 )
  {
    v7 = *v3;
    *(_DWORD *)(a1 + 8) = 32;
    *(_QWORD *)a1 = v7;
    return a1;
  }
  else if ( v4 > 0x20 )
  {
    v8 = *(_QWORD *)v3;
    *(_DWORD *)(a1 + 8) = 64;
    *(_QWORD *)a1 = v8;
    return a1;
  }
  else
  {
    if ( v4 == 8 )
    {
      v5 = *(unsigned __int8 *)v3;
      *(_DWORD *)(a1 + 8) = 8;
    }
    else
    {
      v5 = *(unsigned __int16 *)v3;
      *(_DWORD *)(a1 + 8) = 16;
    }
    *(_QWORD *)a1 = v5;
    return a1;
  }
}
