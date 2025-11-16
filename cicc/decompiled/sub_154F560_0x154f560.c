// Function: sub_154F560
// Address: 0x154f560
//
__int64 __fastcall sub_154F560(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  _BYTE *v5; // rax
  __int64 v6; // r14
  char *v7; // rax
  __int64 v8; // rdx
  const char *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned int v13; // ebx
  int v14; // r14d
  __int64 v15; // rdx
  __int64 v16; // rcx
  _BYTE *v17; // rsi
  __int64 v18; // rdi
  _WORD *v19; // rdx
  __int64 v21; // rdi
  _WORD *v22; // rdx
  int v23; // eax
  __int64 v24; // rdi
  int v25; // r15d
  _BYTE *v26; // rax
  _QWORD *v27; // rdx

  v4 = *a1;
  v5 = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v4 + 16) )
  {
    sub_16E7DE0(v4, 33);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v5 + 1;
    *v5 = 33;
  }
  v6 = *a1;
  v7 = (char *)sub_161F640(a2);
  v9 = (const char *)v8;
  sub_154A520(v7, v8, v6);
  v11 = *a1;
  v12 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v12) <= 4 )
  {
    v9 = " = !{";
    sub_16E7EE0(v11, " = !{", 5);
  }
  else
  {
    *(_DWORD *)v12 = 555760928;
    *(_BYTE *)(v12 + 4) = 123;
    *(_QWORD *)(v11 + 24) += 5LL;
  }
  v13 = 0;
  v14 = sub_161F520(a2, v9, v12, v10);
  if ( v14 )
  {
    while ( 1 )
    {
      v17 = (_BYTE *)sub_161F530(a2, v13);
      if ( *v17 == 6 )
      {
        sub_15499D0(*a1, (__int64)v17);
      }
      else
      {
        v23 = sub_154F490(a1[4], (__int64)v17, v15, v16);
        v24 = *a1;
        v25 = v23;
        if ( v23 == -1 )
        {
          v27 = *(_QWORD **)(v24 + 24);
          if ( *(_QWORD *)(v24 + 16) - (_QWORD)v27 <= 7u )
          {
            sub_16E7EE0(v24, "<badref>", 8);
          }
          else
          {
            *v27 = 0x3E6665726461623CLL;
            *(_QWORD *)(v24 + 24) += 8LL;
          }
        }
        else
        {
          v26 = *(_BYTE **)(v24 + 24);
          if ( (unsigned __int64)v26 >= *(_QWORD *)(v24 + 16) )
          {
            v24 = sub_16E7DE0(v24, 33);
          }
          else
          {
            *(_QWORD *)(v24 + 24) = v26 + 1;
            *v26 = 33;
          }
          sub_16E7AB0(v24, v25);
        }
      }
      if ( v14 == ++v13 )
        break;
      v21 = *a1;
      v22 = *(_WORD **)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v22 <= 1u )
      {
        sub_16E7EE0(v21, ", ", 2);
      }
      else
      {
        *v22 = 8236;
        *(_QWORD *)(v21 + 24) += 2LL;
      }
    }
  }
  v18 = *a1;
  v19 = *(_WORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v19 <= 1u )
    return sub_16E7EE0(v18, "}\n", 2);
  *v19 = 2685;
  *(_QWORD *)(v18 + 24) += 2LL;
  return 2685;
}
