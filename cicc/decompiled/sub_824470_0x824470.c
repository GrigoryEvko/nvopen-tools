// Function: sub_824470
// Address: 0x824470
//
__int64 __fastcall sub_824470(unsigned __int8 *a1, __int64 a2)
{
  const char *v2; // r12
  FILE *v3; // rax
  FILE *v4; // r13
  unsigned __int8 v5; // al
  unsigned __int8 v6; // r14
  unsigned __int8 v7; // bl
  bool v8; // cc
  char *v9; // r15
  _QWORD *v10; // r14
  unsigned int v11; // r12d
  char *v13; // r8
  _QWORD *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // [rsp+8h] [rbp-48h]
  int v22[14]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (const char *)a2;
  v3 = sub_7244D0((char *)a2, "rb", v22);
  if ( v3 )
  {
    v4 = v3;
    v5 = sub_8241D0(v3);
    v6 = *a1;
    if ( v5 && v6 == 3 )
    {
      *a1 = v5;
      v11 = 1;
    }
    else if ( v5 == v6 )
    {
      v11 = 1;
    }
    else
    {
      if ( v6 == 2 )
      {
        v7 = 9;
        v8 = v5 <= 2u;
        if ( v5 == 2 )
        {
          v9 = sub_67C860(3082);
          goto LABEL_12;
        }
        goto LABEL_9;
      }
      if ( v6 != 3 )
        goto LABEL_42;
      v7 = 4;
      if ( v5 <= 1u )
      {
        v8 = v5 <= 2u;
LABEL_9:
        if ( v8 )
        {
          if ( v5 )
            v9 = sub_67C860(3081);
          else
            v9 = sub_67C860(3080);
LABEL_12:
          if ( v6 == 2 )
            v13 = sub_67C860(3082);
          else
            v13 = sub_67C860(3148);
          v21 = (__int64)v13;
          v10 = sub_67D610(0xC05u, dword_4F07508, v7);
          sub_67EFD0((__int64)v10, v21);
          sub_67EFD0((__int64)v10, (__int64)v9);
          sub_67DC60(v10, 3078, a2);
          v11 = 0;
          sub_685910((__int64)v10, (FILE *)0xC06);
          goto LABEL_16;
        }
        if ( v5 == 3 )
        {
          v9 = sub_67C860(3148);
          goto LABEL_12;
        }
        sub_67C860(3084);
LABEL_42:
        sub_721090();
      }
      v11 = 0;
    }
LABEL_16:
    fclose(v4);
    return v11;
  }
  if ( (v22[0] & 2) != 0 )
    sub_685AD0(8u, 3074, a2, v22);
  v14 = sub_67D610(0xCE0u, dword_4F07508, 9u);
  sub_67EFD0((__int64)v14, a2);
  v15 = *(_QWORD *)qword_4D048F0;
  v16 = *(_QWORD *)qword_4D048F0 + 16LL * (unsigned int)(*(_DWORD *)(qword_4D048F0 + 8) + 1);
  if ( *(_QWORD *)qword_4D048F0 != v16 )
  {
    do
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)v15;
        if ( *(_QWORD *)v15 )
        {
          a2 = (__int64)v2;
          if ( !strcmp(*(const char **)(v15 + 8), v2) )
            break;
        }
        v15 += 16;
        if ( v16 == v15 )
          goto LABEL_28;
      }
      a2 = 3297;
      v15 += 16;
      sub_67DC60(v14, 3297, v17);
    }
    while ( v16 != v15 );
  }
LABEL_28:
  v18 = *qword_4D048E0;
  v19 = *qword_4D048E0 + 16LL * (unsigned int)(*((_DWORD *)qword_4D048E0 + 2) + 1);
  if ( v19 != *qword_4D048E0 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)v18;
        if ( *(_QWORD *)v18 )
        {
          a2 = (__int64)v2;
          if ( !strcmp(*(const char **)(v18 + 8), v2) )
            break;
        }
        v18 += 16;
        if ( v19 == v18 )
          goto LABEL_34;
      }
      a2 = 3298;
      v18 += 16;
      sub_67DC60(v14, 3298, v20);
    }
    while ( v19 != v18 );
  }
LABEL_34:
  v11 = 0;
  sub_685910((__int64)v14, (FILE *)a2);
  return v11;
}
