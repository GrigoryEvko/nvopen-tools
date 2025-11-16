// Function: sub_1BE31F0
// Address: 0x1be31f0
//
_BYTE *__fastcall sub_1BE31F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rdx
  _WORD *v8; // rdx
  _BYTE *result; // rax
  __int64 v10; // rdx
  int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax

  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v6 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 10;
    v6 = a2;
    *(_WORD *)v5 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v6);
  v7 = *(_QWORD *)(v6 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 16) - v7) <= 6 )
  {
    sub_16E7EE0(v6, "\"BLEND ", 7u);
  }
  else
  {
    *(_DWORD *)v7 = 1162625570;
    *(_WORD *)(v7 + 4) = 17486;
    *(_BYTE *)(v7 + 6) = 32;
    *(_QWORD *)(v6 + 24) += 7LL;
  }
  sub_15537D0(*(_QWORD *)(a1 + 40), a2, 0, 0);
  v8 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v8 > 1u )
  {
    *v8 = 15648;
    result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 2LL);
    *(_QWORD *)(a2 + 24) = result;
    v10 = *(_QWORD *)(a1 + 48);
    if ( v10 )
      goto LABEL_7;
LABEL_32:
    if ( *(_BYTE **)(a2 + 16) == result )
    {
      sub_16E7EE0(a2, " ", 1u);
    }
    else
    {
      *result = 32;
      ++*(_QWORD *)(a2 + 24);
    }
    v20 = *(_QWORD *)(a1 + 40);
    if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
      v21 = *(__int64 **)(v20 - 8);
    else
      v21 = (__int64 *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
    sub_15537D0(*v21, a2, 0, 0);
    result = *(_BYTE **)(a2 + 24);
    goto LABEL_27;
  }
  sub_16E7EE0(a2, " =", 2u);
  v10 = *(_QWORD *)(a1 + 48);
  result = *(_BYTE **)(a2 + 24);
  if ( !v10 )
    goto LABEL_32;
LABEL_7:
  v11 = *(_DWORD *)(v10 + 48);
  if ( v11 )
  {
    v12 = 0;
    v13 = 8LL * (unsigned int)(v11 - 1);
    while ( 1 )
    {
      if ( *(_BYTE **)(a2 + 16) == result )
      {
        sub_16E7EE0(a2, " ", 1u);
      }
      else
      {
        *result = 32;
        ++*(_QWORD *)(a2 + 24);
      }
      v19 = *(_QWORD *)(a1 + 40);
      v14 = (*(_BYTE *)(v19 + 23) & 0x40) != 0 ? *(_QWORD *)(v19 - 8) : v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF);
      sub_15537D0(*(_QWORD *)(v14 + 3 * v12), a2, 0, 0);
      v15 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v15 )
      {
        sub_16E7EE0(a2, "/", 1u);
        v16 = *(_QWORD *)(a2 + 24);
      }
      else
      {
        *v15 = 47;
        v16 = *(_QWORD *)(a2 + 24) + 1LL;
        *(_QWORD *)(a2 + 24) = v16;
      }
      v17 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 40LL) + v12);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v16) <= 2 )
      {
        v18 = sub_16E7EE0(a2, "%vp", 3u);
      }
      else
      {
        *(_BYTE *)(v16 + 2) = 112;
        v18 = a2;
        *(_WORD *)v16 = 30245;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      sub_16E7AB0(v18, (unsigned __int16)v17);
      result = *(_BYTE **)(a2 + 24);
      if ( v13 == v12 )
        break;
      v12 += 8;
    }
  }
LABEL_27:
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 2u )
    return (_BYTE *)sub_16E7EE0(a2, "\\l\"", 3u);
  result[2] = 34;
  *(_WORD *)result = 27740;
  *(_QWORD *)(a2 + 24) += 3LL;
  return result;
}
