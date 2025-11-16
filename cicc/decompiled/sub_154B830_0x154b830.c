// Function: sub_154B830
// Address: 0x154b830
//
void __fastcall sub_154B830(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rax
  size_t v5; // rdx
  size_t v6; // r14
  const void *v7; // r15
  const void *v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rax
  const char *v11; // r13
  size_t v12; // rdx
  size_t v13; // rbx
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax

  v2 = *(_QWORD *)(a2 + 48);
  if ( v2 )
  {
    if ( *(_BYTE *)(a2 + 16) == 3 )
    {
      v16 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(a1 + 16) )
      {
        sub_16E7DE0(a1, 44);
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v16 + 1;
        *v16 = 44;
      }
    }
    v3 = *(_QWORD *)(a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v3) <= 6 )
    {
      sub_16E7EE0(a1, " comdat", 7);
    }
    else
    {
      *(_DWORD *)v3 = 1836016416;
      *(_WORD *)(v3 + 4) = 24932;
      *(_BYTE *)(v3 + 6) = 116;
      *(_QWORD *)(a1 + 24) += 7LL;
    }
    v4 = sub_1580C70(v2);
    v6 = v5;
    v7 = (const void *)v4;
    v8 = (const void *)sub_1649960(a2);
    if ( v6 != v9 || v6 && memcmp(v8, v7, v6) )
    {
      v10 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 16) )
      {
        sub_16E7DE0(a1, 40);
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v10 + 1;
        *v10 = 40;
      }
      v11 = (const char *)sub_1580C70(v2);
      v13 = v12;
      v14 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(a1 + 16) )
      {
        sub_16E7DE0(a1, 36);
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v14 + 1;
        *v14 = 36;
      }
      sub_154B650(a1, v11, v13);
      v15 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(a1 + 16) )
      {
        sub_16E7DE0(a1, 41);
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v15 + 1;
        *v15 = 41;
      }
    }
  }
}
