// Function: sub_24511A0
// Address: 0x24511a0
//
char __fastcall sub_24511A0(__int64 a1, __int64 a2, __int64 a3, void *a4, size_t a5)
{
  __int16 v8; // ax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  char *v13; // rax
  size_t v14; // rdx

  LOBYTE(v8) = sub_ED2B40(a3, *(_QWORD *)a1);
  if ( (_BYTE)v8 )
  {
    if ( *(_DWORD *)(a1 + 100) == 1 && *(_BYTE *)(a1 + 144) )
    {
      v13 = (char *)sub_BD5D20(a2);
      v12 = sub_BAA410(*(_QWORD *)a1, v13, v14);
    }
    else
    {
      v12 = sub_BAA410(*(_QWORD *)a1, a4, a5);
    }
  }
  else
  {
    if ( *(_DWORD *)(a1 + 100) != 3 )
      return v8;
    v9 = sub_BAA410(*(_QWORD *)a1, a4, a5);
    *(_DWORD *)(v9 + 8) = 3;
    v12 = v9;
  }
  LOBYTE(v8) = (unsigned __int8)sub_B2F990(a2, v12, v10, v11);
  if ( *(_DWORD *)(a1 + 100) == 1 )
  {
    LOBYTE(v8) = *(_BYTE *)(a2 + 32) & 0xF;
    if ( (_BYTE)v8 == 8 )
    {
      v8 = *(_WORD *)(a2 + 32) & 0xBCC0 | 0x4007;
      *(_WORD *)(a2 + 32) = v8;
    }
  }
  return v8;
}
