// Function: sub_AA5EE0
// Address: 0xaa5ee0
//
__int64 __fastcall sub_AA5EE0(__int64 a1)
{
  unsigned __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int8 v5; // al
  _QWORD *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // [rsp-28h] [rbp-28h]

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
    goto LABEL_20;
  if ( !v1 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 > 0xA )
LABEL_20:
    BUG();
  if ( (*(_BYTE *)(v1 - 17) & 0x20) != 0 )
  {
    v2 = sub_B91C10(v1 - 24, 24);
    v3 = v2;
    if ( v2 )
    {
      v4 = v2 - 16;
      v5 = *(_BYTE *)(v2 - 16);
      v6 = (v5 & 2) != 0 ? *(_QWORD **)(v3 - 32) : (_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF));
      v7 = sub_B91420(*v6, 24);
      if ( v8 == 18
        && !(*(_QWORD *)v7 ^ 0x6165685F706F6F6CLL | *(_QWORD *)(v7 + 8) ^ 0x676965775F726564LL)
        && *(_WORD *)(v7 + 16) == 29800 )
      {
        v10 = *(_BYTE *)(v3 - 16);
        if ( (v10 & 2) != 0 )
          v11 = *(_QWORD *)(v3 - 32);
        else
          v11 = v4 - 8LL * ((v10 >> 2) & 0xF);
        v12 = *(_QWORD *)(*(_QWORD *)(v11 + 8) + 136LL);
        v13 = *(_QWORD *)(v12 + 24);
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
          return *(_QWORD *)v13;
        return v13;
      }
    }
  }
  return v14;
}
