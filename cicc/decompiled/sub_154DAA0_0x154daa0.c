// Function: sub_154DAA0
// Address: 0x154daa0
//
_BYTE *__fastcall sub_154DAA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // ebx
  unsigned __int8 v6; // si
  __int64 v7; // rax
  __int64 v8; // rax
  const char *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // r15
  __int64 *v14; // rbx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  int v19; // r8d
  unsigned int v20; // edx
  __int64 *v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  const char *v28; // r12
  size_t v29; // rdx
  size_t v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rdi

  switch ( *(_BYTE *)(a2 + 8) )
  {
    case 0:
      v10 = "void";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 1:
      v10 = "half";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 2:
      v10 = "float";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 3:
      v10 = "double";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 4:
      v10 = "x86_fp80";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 5:
      v10 = "fp128";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 6:
      v10 = "ppc_fp128";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 7:
      v10 = "label";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 8:
      v10 = "metadata";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 9:
      v10 = "x86_mmx";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 0xA:
      v10 = "token";
      return (_BYTE *)sub_1263B40(a3, v10);
    case 0xB:
      v11 = sub_1549FC0(a3, 0x69u);
      return (_BYTE *)sub_16E7A90(v11, *(_DWORD *)(a2 + 8) >> 8);
    case 0xC:
      sub_154DAA0(a1, **(_QWORD **)(a2 + 16), a3);
      sub_1263B40(a3, " (");
      v12 = *(_QWORD *)(a2 + 16);
      v13 = (__int64 *)(v12 + 8);
      v14 = (__int64 *)(v12 + 8LL * *(unsigned int *)(a2 + 12));
      if ( v14 != (__int64 *)(v12 + 8) )
      {
        while ( 1 )
        {
          v15 = *v13++;
          sub_154DAA0(a1, v15, a3);
          if ( v14 == v13 )
            break;
          if ( v13 != (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL) )
            sub_1263B40(a3, ", ");
        }
      }
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        if ( *(_DWORD *)(a2 + 12) != 1 )
          sub_1263B40(a3, ", ");
        sub_1263B40(a3, "...");
      }
      v6 = 41;
      return (_BYTE *)sub_1549FC0(a3, v6);
    case 0xD:
      if ( (*(_BYTE *)(a2 + 9) & 4) != 0 )
        return (_BYTE *)sub_154DEB0(a1, a2, a3);
      sub_1643640(a2);
      if ( v16 )
      {
        v28 = (const char *)sub_1643640(a2);
        v30 = v29;
        sub_1549FC0(a3, 0x25u);
        return sub_154B650(a3, v28, v30);
      }
      sub_154D780(a1);
      v17 = *(unsigned int *)(a1 + 160);
      if ( !(_DWORD)v17 )
        goto LABEL_38;
      v18 = *(_QWORD *)(a1 + 144);
      v19 = 1;
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( a2 == *v21 )
        goto LABEL_21;
      break;
    case 0xE:
      v24 = sub_1549FC0(a3, 0x5Bu);
      v25 = sub_16E7A90(v24, *(_QWORD *)(a2 + 32));
      sub_1263B40(v25, " x ");
      sub_154DAA0(a1, *(_QWORD *)(a2 + 24), a3);
      v6 = 93;
      return (_BYTE *)sub_1549FC0(a3, v6);
    case 0xF:
      sub_154DAA0(a1, *(_QWORD *)(a2 + 24), a3);
      v5 = *(_DWORD *)(a2 + 8) >> 8;
      if ( v5 )
      {
        v26 = sub_1263B40(a3, " addrspace(");
        v27 = sub_16E7A90(v26, v5);
        sub_1549FC0(v27, 0x29u);
      }
      v6 = 42;
      return (_BYTE *)sub_1549FC0(a3, v6);
    case 0x10:
      v7 = sub_1263B40(a3, "<");
      v8 = sub_16E7A90(v7, *(_QWORD *)(a2 + 32));
      sub_1263B40(v8, " x ");
      sub_154DAA0(a1, *(_QWORD *)(a2 + 24), a3);
      v6 = 62;
      return (_BYTE *)sub_1549FC0(a3, v6);
  }
  while ( v22 != -8 )
  {
    v20 = (v17 - 1) & (v19 + v20);
    v21 = (__int64 *)(v18 + 16LL * v20);
    v22 = *v21;
    if ( a2 == *v21 )
    {
LABEL_21:
      if ( v21 != (__int64 *)(v18 + 16 * v17) )
      {
        v23 = sub_1549FC0(a3, 0x25u);
        return (_BYTE *)sub_16E7A90(v23, *((unsigned int *)v21 + 2));
      }
      break;
    }
    ++v19;
  }
LABEL_38:
  v31 = sub_1263B40(a3, "%\"type ");
  v32 = sub_16E7B40(v31, a2);
  return (_BYTE *)sub_1549FC0(v32, 0x22u);
}
