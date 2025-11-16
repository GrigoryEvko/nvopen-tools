// Function: sub_B90960
// Address: 0xb90960
//
__int64 __fastcall sub_B90960(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 v3; // r13
  bool v4; // zf
  __int64 v5; // rax
  unsigned __int8 v6; // al
  _QWORD *v7; // rdx
  __int64 result; // rax
  int v9; // eax
  int v10; // eax

  v2 = a2 - 16;
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_BYTE *)a2 == 16;
    *(_QWORD *)a1 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( v4 )
    {
      v10 = *(_DWORD *)(a2 + 16);
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = v10;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
      *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
    }
    v5 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    v4 = *(_BYTE *)a2 == 16;
    *(_QWORD *)a1 = *(_QWORD *)(a2 - 8LL * ((v3 >> 2) & 0xF));
    if ( v4 )
    {
      v9 = *(_DWORD *)(a2 + 16);
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = v9;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
      *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
    }
    v5 = v2 - 8LL * ((v3 >> 2) & 0xF);
  }
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(v5 + 8);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 24);
  *(_DWORD *)(a1 + 40) = sub_AF18D0(a2);
  *(_DWORD *)(a1 + 44) = *(_DWORD *)(a2 + 20);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 24LL);
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    *(_QWORD *)(a1 + 64) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 40LL);
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 48LL);
    v7 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v7 = (_QWORD *)(v2 - 8LL * ((v6 >> 2) & 0xF));
    *(_QWORD *)(a1 + 48) = v7[3];
    *(_QWORD *)(a1 + 56) = v7[4];
    *(_QWORD *)(a1 + 64) = v7[5];
    *(_QWORD *)(a1 + 72) = v7[6];
  }
  result = v7[7];
  *(_QWORD *)(a1 + 80) = result;
  return result;
}
