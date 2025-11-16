// Function: sub_37BFC10
// Address: 0x37bfc10
//
__int64 __fastcall sub_37BFC10(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rdx
  __int64 v3; // rcx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // edx
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r13
  int v11; // eax
  unsigned int v12; // ecx
  unsigned int v13; // eax
  int v14; // r14d
  unsigned int v15; // eax
  char v16; // al
  int v17; // esi
  int v18; // r15d
  unsigned int v19; // edx
  unsigned int v20; // eax
  int v21; // [rsp+Ch] [rbp-94h]
  _QWORD v22[6]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v23[12]; // [rsp+40h] [rbp-60h] BYREF

  v1 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( v1 )
  {
    v12 = 4 * v1;
    v2 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)(4 * v1) < 0x40 )
      v12 = 64;
    if ( (unsigned int)v2 <= v12 )
      goto LABEL_4;
    v13 = v1 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v13, v13);
      v14 = 1 << (33 - (v13 ^ 0x1F));
      if ( v14 < 64 )
        v14 = 64;
      if ( v14 == (_DWORD)v2 )
        goto LABEL_25;
    }
    else
    {
      v14 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * (unsigned int)v2, 8);
    v15 = sub_37B8280(v14);
    *(_DWORD *)(a1 + 56) = v15;
    if ( !v15 )
      goto LABEL_44;
    *(_QWORD *)(a1 + 40) = sub_C7D670(16LL * v15, 8);
LABEL_25:
    sub_37BFB80(a1 + 32);
    goto LABEL_7;
  }
  if ( *(_DWORD *)(a1 + 52) )
  {
    v2 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v2 <= 0x40 )
    {
LABEL_4:
      v3 = unk_5051170;
      v4 = *(_QWORD **)(a1 + 40);
      for ( i = &v4[2 * v2]; i != v4; v4 += 2 )
        *v4 = v3;
      goto LABEL_6;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * (unsigned int)v2, 8);
    *(_DWORD *)(a1 + 56) = 0;
LABEL_44:
    *(_QWORD *)(a1 + 40) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 48) = 0;
  }
LABEL_7:
  v6 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  if ( v6 || (result = *(unsigned int *)(a1 + 84), (_DWORD)result) )
  {
    v8 = *(_QWORD *)(a1 + 72);
    result = (unsigned int)(4 * v6);
    v9 = 48LL * *(unsigned int *)(a1 + 88);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v10 = v8 + v9;
    if ( *(_DWORD *)(a1 + 88) <= (unsigned int)result )
    {
      for ( ; v8 != v10; *(_DWORD *)(v8 - 48) = result )
      {
        v11 = *(_DWORD *)v8;
        *(_QWORD *)(v8 + 16) = 0;
        v8 += 48;
        result = v11 & 0xFFF00000 | 0x15;
      }
      goto LABEL_14;
    }
    v22[2] = 0;
    v22[0] = 21;
    v23[0] = 22;
    v23[2] = 0;
    do
    {
      if ( (unsigned __int8)(*(_BYTE *)v8 - 21) > 1u )
      {
        v21 = v6;
        v16 = sub_2EAB6C0(v8, (char *)v22);
        v6 = v21;
        if ( !v16 && (unsigned __int8)(*(_BYTE *)v8 - 21) > 1u )
        {
          sub_2EAB6C0(v8, (char *)v23);
          v6 = v21;
        }
      }
      v8 += 48;
    }
    while ( v8 != v10 );
    v17 = *(_DWORD *)(a1 + 88);
    if ( v6 )
    {
      v18 = 64;
      v19 = v6 - 1;
      if ( v19 )
      {
        _BitScanReverse(&v20, v19);
        v18 = 1 << (33 - (v20 ^ 0x1F));
        if ( v18 < 64 )
          v18 = 64;
      }
      if ( v18 == v17 )
        return (__int64)sub_37BFBC0(a1 + 64);
      sub_C7D6A0(*(_QWORD *)(a1 + 72), v9, 8);
      result = sub_37B8280(v18);
      *(_DWORD *)(a1 + 88) = result;
      if ( (_DWORD)result )
      {
        *(_QWORD *)(a1 + 72) = sub_C7D670(48LL * (unsigned int)result, 8);
        return (__int64)sub_37BFBC0(a1 + 64);
      }
    }
    else
    {
      if ( !v17 )
        return (__int64)sub_37BFBC0(a1 + 64);
      result = sub_C7D6A0(*(_QWORD *)(a1 + 72), v9, 8);
      *(_DWORD *)(a1 + 88) = 0;
    }
    *(_QWORD *)(a1 + 72) = 0;
LABEL_14:
    *(_QWORD *)(a1 + 80) = 0;
  }
  return result;
}
