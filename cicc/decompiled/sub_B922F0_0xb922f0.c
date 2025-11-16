// Function: sub_B922F0
// Address: 0xb922f0
//
__int64 __fastcall sub_B922F0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r14
  unsigned __int8 v3; // r13
  __int64 v4; // rax
  bool v5; // zf
  _BYTE *v6; // rax
  int v7; // ecx
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 result; // rax
  __int64 v13; // rax
  unsigned __int16 v14; // r15
  __int64 v15; // r13
  _QWORD *v16; // rcx
  __int16 v17; // r15
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // [rsp+8h] [rbp-38h]

  v2 = (_BYTE *)(a2 - 16);
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
  {
    *(_QWORD *)a1 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    *(_QWORD *)(a1 + 8) = sub_AF5140(a2, 2u);
    v4 = sub_AF5140(a2, 3u);
    v5 = *(_BYTE *)a2 == 16;
    *(_QWORD *)(a1 + 16) = v4;
    if ( v5 )
    {
      v21 = *(_DWORD *)(a2 + 16);
      v7 = *(_DWORD *)(a2 + 20);
      *(_QWORD *)(a1 + 24) = a2;
      *(_DWORD *)(a1 + 32) = v21;
    }
    else
    {
      v6 = sub_A17150(v2);
      v7 = *(_DWORD *)(a2 + 20);
      *(_QWORD *)(a1 + 24) = *(_QWORD *)v6;
      *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 16);
    }
    v8 = *(_QWORD *)(a2 - 32);
    v9 = *(_DWORD *)(a2 - 24);
    *(_DWORD *)(a1 + 48) = v7;
    *(_QWORD *)(a1 + 40) = *(_QWORD *)(v8 + 32);
    if ( v9 <= 8 )
    {
      v22 = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = v22;
    }
    else
    {
      *(_QWORD *)(a1 + 56) = *((_QWORD *)sub_A17150(v2) + 8);
      *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 24);
    }
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 32);
    v10 = *(_DWORD *)(a2 - 24);
    *(_QWORD *)(a1 + 80) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 40LL);
    if ( v10 <= 9 )
      *(_QWORD *)(a1 + 88) = 0;
    else
      *(_QWORD *)(a1 + 88) = *((_QWORD *)sub_A17150(v2) + 9);
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 48LL);
    v11 = *(_DWORD *)(a2 - 24);
    *(_QWORD *)(a1 + 104) = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 56LL);
    if ( v11 <= 0xA )
      goto LABEL_9;
    goto LABEL_17;
  }
  *(_QWORD *)a1 = *(_QWORD *)&v2[-8 * ((v3 >> 2) & 0xF) + 8];
  *(_QWORD *)(a1 + 8) = sub_AF5140(a2, 2u);
  v13 = sub_AF5140(a2, 3u);
  v5 = *(_BYTE *)a2 == 16;
  *(_QWORD *)(a1 + 16) = v13;
  if ( v5 )
  {
    v20 = *(_DWORD *)(a2 + 16);
    *(_QWORD *)(a1 + 24) = a2;
    *(_DWORD *)(a1 + 32) = v20;
  }
  else
  {
    *(_QWORD *)(a1 + 24) = *(_QWORD *)sub_A17150(v2);
    *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 16);
  }
  v14 = *(_WORD *)(a2 - 16);
  *(_DWORD *)(a1 + 48) = *(_DWORD *)(a2 + 20);
  v15 = 8LL * ((v3 >> 2) & 0xF);
  v16 = &v2[-v15];
  v17 = (v14 >> 6) & 0xF;
  *(_QWORD *)(a1 + 40) = *(_QWORD *)&v2[-v15 + 32];
  if ( (unsigned __int8)v17 <= 8u )
  {
    v23 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 56) = 0;
    v11 = (unsigned __int8)v17;
    *(_QWORD *)(a1 + 64) = v23;
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a1 + 80) = v16[5];
  }
  else
  {
    v24 = &v2[-v15];
    v11 = (unsigned __int8)v17;
    v18 = sub_A17150(v2);
    v16 = v24;
    *(_QWORD *)(a1 + 56) = *((_QWORD *)v18 + 8);
    *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a1 + 80) = v24[5];
    if ( (unsigned __int8)v17 > 9u )
    {
      v19 = sub_A17150(v2);
      v16 = v24;
      *(_QWORD *)(a1 + 88) = *((_QWORD *)v19 + 9);
      goto LABEL_16;
    }
  }
  *(_QWORD *)(a1 + 88) = 0;
LABEL_16:
  *(_QWORD *)(a1 + 96) = v16[6];
  *(_QWORD *)(a1 + 104) = v16[7];
  if ( v11 <= 0xA )
  {
LABEL_9:
    *(_QWORD *)(a1 + 112) = 0;
LABEL_10:
    *(_QWORD *)(a1 + 120) = 0;
    result = 0;
    goto LABEL_20;
  }
LABEL_17:
  *(_QWORD *)(a1 + 112) = *((_QWORD *)sub_A17150(v2) + 10);
  if ( v11 <= 0xB )
    goto LABEL_10;
  *(_QWORD *)(a1 + 120) = *((_QWORD *)sub_A17150(v2) + 11);
  if ( v11 <= 0xC )
    result = 0;
  else
    result = sub_AF5140(a2, 0xCu);
LABEL_20:
  *(_QWORD *)(a1 + 128) = result;
  return result;
}
