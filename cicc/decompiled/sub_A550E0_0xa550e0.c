// Function: sub_A550E0
// Address: 0xa550e0
//
void __fastcall sub_A550E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  const void *v7; // r14
  const void *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rax
  unsigned __int8 *v11; // r13
  size_t v12; // rdx
  size_t v13; // rbx
  _BYTE *v14; // rax
  _BYTE *v15; // rax

  v2 = *(_QWORD *)(a2 + 48);
  if ( v2 )
  {
    if ( *(_BYTE *)a2 == 3 )
    {
      v15 = *(_BYTE **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) <= (unsigned __int64)v15 )
      {
        sub_CB5D20(a1, 44);
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v15 + 1;
        *v15 = 44;
      }
    }
    v3 = *(_QWORD *)(a1 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v3) <= 6 )
    {
      sub_CB6200(a1, " comdat", 7);
    }
    else
    {
      *(_DWORD *)v3 = 1836016416;
      *(_WORD *)(v3 + 4) = 24932;
      *(_BYTE *)(v3 + 6) = 116;
      *(_QWORD *)(a1 + 32) += 7LL;
    }
    v4 = sub_AA8810(v2);
    v6 = v5;
    v7 = (const void *)v4;
    v8 = (const void *)sub_BD5D20(a2);
    if ( v9 != v6 || v9 && memcmp(v8, v7, v9) )
    {
      v10 = *(_BYTE **)(a1 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 24) )
      {
        sub_CB5D20(a1, 40);
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v10 + 1;
        *v10 = 40;
      }
      v11 = (unsigned __int8 *)sub_AA8810(v2);
      v13 = v12;
      sub_A51310(a1, 0x24u);
      sub_A54F00(a1, v11, v13);
      v14 = *(_BYTE **)(a1 + 32);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(a1 + 24) )
      {
        sub_CB5D20(a1, 41);
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v14 + 1;
        *v14 = 41;
      }
    }
  }
}
