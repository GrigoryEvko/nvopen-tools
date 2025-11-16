// Function: sub_B907E0
// Address: 0xb907e0
//
__int64 __fastcall sub_B907E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int8 v3; // al
  bool v4; // zf
  int v5; // eax
  _QWORD *v6; // rdx
  int v7; // eax
  unsigned __int8 v8; // al
  __int64 v9; // r13
  __int64 result; // rax

  v2 = a2 - 16;
  *(_DWORD *)a1 = (unsigned __int16)sub_AF18C0(a2);
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_BYTE *)a2 == 16;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( v4 )
    {
      v5 = *(_DWORD *)(a2 + 16);
      *(_QWORD *)(a1 + 16) = a2;
    }
    else
    {
      *(_QWORD *)(a1 + 16) = **(_QWORD **)(a2 - 32);
      v5 = *(_DWORD *)(a2 + 16);
    }
    *(_DWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    v6 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v6 = (_QWORD *)(v2 - 8LL * ((v3 >> 2) & 0xF));
    v4 = *(_BYTE *)a2 == 16;
    *(_QWORD *)(a1 + 8) = v6[2];
    if ( v4 )
    {
      v7 = *(_DWORD *)(a2 + 16);
      *(_QWORD *)(a1 + 16) = a2;
    }
    else
    {
      *(_QWORD *)(a1 + 16) = *v6;
      v7 = *(_DWORD *)(a2 + 16);
    }
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 32) = v6[1];
  }
  *(_QWORD *)(a1 + 40) = v6[3];
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 64) = sub_AF18D0(a2);
  *(_QWORD *)(a1 + 68) = *(_QWORD *)(a2 + 44);
  *(_QWORD *)(a1 + 76) = sub_AF2E40(a2);
  *(_DWORD *)(a1 + 84) = *(_DWORD *)(a2 + 20);
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 88) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    v9 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    v9 = v2 - 8LL * ((v8 >> 2) & 0xF);
    *(_QWORD *)(a1 + 88) = *(_QWORD *)(v9 + 32);
  }
  result = *(_QWORD *)(v9 + 40);
  *(_QWORD *)(a1 + 96) = result;
  return result;
}
