// Function: sub_A5D2A0
// Address: 0xa5d2a0
//
__int64 __fastcall sub_A5D2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int8 v10; // al
  __int64 *v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rbx
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  char v22; // [rsp+18h] [rbp-38h]
  char *v23; // [rsp+20h] [rbp-30h]
  __int64 v24; // [rsp+28h] [rbp-28h]

  sub_904010(a1, "!DIDerivedType(");
  v24 = a3;
  v21 = a1;
  v4 = a2 - 16;
  v23 = ", ";
  v22 = 1;
  sub_A53560(&v21, a2);
  v5 = sub_A547D0(a2, 2);
  sub_A53660(&v21, "name", 4u, v5, v6, 1);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a2 - 32);
  else
    v8 = v4 - 8LL * ((v7 >> 2) & 0xF);
  sub_A5CC00((__int64)&v21, "scope", 5u, *(_QWORD *)(v8 + 8), 1);
  v9 = a2;
  if ( *(_BYTE *)a2 != 16 )
  {
    v10 = *(_BYTE *)(a2 - 16);
    if ( (v10 & 2) != 0 )
      v11 = *(__int64 **)(a2 - 32);
    else
      v11 = (__int64 *)(v4 - 8LL * ((v10 >> 2) & 0xF));
    v9 = *v11;
  }
  sub_A5CC00((__int64)&v21, "file", 4u, v9, 1);
  sub_A537C0((__int64)&v21, "line", 4u, *(_DWORD *)(a2 + 16), 1);
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v4 - 8LL * ((v12 >> 2) & 0xF);
  sub_A5CC00((__int64)&v21, "baseType", 8u, *(_QWORD *)(v13 + 24), 0);
  sub_A539C0((__int64)&v21, "size", 4u, *(_QWORD *)(a2 + 24));
  v14 = sub_AF18D0(a2);
  sub_A537C0((__int64)&v21, "align", 5u, v14, 1);
  sub_A539C0((__int64)&v21, "offset", 6u, *(_QWORD *)(a2 + 32));
  sub_A53C60(&v21, "flags", 5u, *(_DWORD *)(a2 + 20));
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a2 - 32);
  else
    v16 = v4 - 8LL * ((v15 >> 2) & 0xF);
  sub_A5CC00((__int64)&v21, "extraData", 9u, *(_QWORD *)(v16 + 32), 1);
  if ( *(_BYTE *)(a2 + 48) )
    sub_A537C0((__int64)&v21, "dwarfAddressSpace", 0x11u, *(_DWORD *)(a2 + 44), 0);
  v17 = *(_BYTE *)(a2 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(_QWORD *)(a2 - 32);
  else
    v18 = v4 - 8LL * ((v17 >> 2) & 0xF);
  sub_A5CC00((__int64)&v21, "annotations", 0xBu, *(_QWORD *)(v18 + 40), 1);
  v20 = sub_AF2E40(a2);
  if ( BYTE4(v20) )
  {
    sub_A537C0((__int64)&v21, "ptrAuthKey", 0xAu, v20 & 0xF, 1);
    sub_A53370((__int64)&v21, "ptrAuthIsAddressDiscriminated", 0x1Du, (v20 & 0x10) != 0, 0);
    sub_A537C0((__int64)&v21, "ptrAuthExtraDiscriminator", 0x19u, (unsigned __int16)((unsigned int)v20 >> 5), 1);
    sub_A53370((__int64)&v21, "ptrAuthIsaPointer", 0x11u, (v20 & 0x200000) != 0, 0);
    sub_A53370((__int64)&v21, "ptrAuthAuthenticatesNullValues", 0x1Eu, (v20 & 0x400000) != 0, 0);
  }
  return sub_904010(a1, ")");
}
