// Function: sub_AF54F0
// Address: 0xaf54f0
//
__int64 __fastcall sub_AF54F0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  unsigned __int8 v6; // al
  _BYTE *v7; // rdx
  unsigned __int8 v8; // al
  _QWORD *v9; // r13
  __int64 result; // rax

  v2 = (_BYTE *)(a2 - 16);
  *(_DWORD *)a1 = (unsigned __int16)sub_AF18C0(a2);
  v3 = sub_AF5140(a2, 2u);
  v4 = *(_BYTE *)a2 == 16;
  *(_QWORD *)(a1 + 8) = v3;
  v5 = a2;
  if ( !v4 )
    v5 = *(_QWORD *)sub_A17150(v2);
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a2 + 16);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    v7 = *(_BYTE **)(a2 - 32);
  }
  else
  {
    v7 = &v2[-8 * ((v6 >> 2) & 0xF)];
    *(_QWORD *)(a1 + 32) = *((_QWORD *)v7 + 1);
  }
  *(_QWORD *)(a1 + 40) = *((_QWORD *)v7 + 3);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 64) = sub_AF18D0(a2);
  *(_DWORD *)(a1 + 68) = *(_DWORD *)(a2 + 20);
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
  {
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
    *(_DWORD *)(a1 + 80) = *(_DWORD *)(a2 + 44);
    *(_QWORD *)(a1 + 88) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 40LL);
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 48LL);
    *(_QWORD *)(a1 + 104) = sub_AF5140(a2, 7u);
    *(_QWORD *)(a1 + 112) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 64LL);
    *(_QWORD *)(a1 + 120) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 72LL);
    *(_QWORD *)(a1 + 128) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 80LL);
    *(_QWORD *)(a1 + 136) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 88LL);
    *(_QWORD *)(a1 + 144) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 96LL);
    *(_QWORD *)(a1 + 152) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 104LL);
    v9 = *(_QWORD **)(a2 - 32);
  }
  else
  {
    v9 = &v2[-8 * ((v8 >> 2) & 0xF)];
    *(_QWORD *)(a1 + 72) = v9[4];
    *(_DWORD *)(a1 + 80) = *(_DWORD *)(a2 + 44);
    *(_QWORD *)(a1 + 88) = v9[5];
    *(_QWORD *)(a1 + 96) = v9[6];
    *(_QWORD *)(a1 + 104) = sub_AF5140(a2, 7u);
    *(_QWORD *)(a1 + 112) = v9[8];
    *(_QWORD *)(a1 + 120) = v9[9];
    *(_QWORD *)(a1 + 128) = v9[10];
    *(_QWORD *)(a1 + 136) = v9[11];
    *(_QWORD *)(a1 + 144) = v9[12];
    *(_QWORD *)(a1 + 152) = v9[13];
  }
  *(_QWORD *)(a1 + 160) = v9[14];
  result = *(unsigned int *)(a2 + 40);
  *(_DWORD *)(a1 + 168) = result;
  return result;
}
