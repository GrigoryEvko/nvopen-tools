// Function: sub_8489F0
// Address: 0x8489f0
//
__int64 __fastcall sub_8489F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // r14
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // r15
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // r12
  char v20; // al
  __int64 v21; // rdx
  _DWORD *v22; // r15
  __int64 v23; // rax
  __int64 j; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int8 v35; // al
  char i; // al
  unsigned __int8 v37; // al
  unsigned __int8 v38; // bl
  __int64 v39; // rax
  unsigned __int8 v40; // al
  __int64 v41; // [rsp+0h] [rbp-1B0h]
  __int64 v42; // [rsp+0h] [rbp-1B0h]
  __int64 v43; // [rsp+8h] [rbp-1A8h]
  __int64 v44; // [rsp+18h] [rbp-198h] BYREF
  __m128i v45[25]; // [rsp+20h] [rbp-190h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a2 + 8);
  result = sub_6E1A20(a1);
  ++*(_DWORD *)(a2 + 28);
  if ( *(_BYTE *)(a2 + 22) )
    return result;
  if ( *(_BYTE *)(a2 + 16) || *(_BYTE *)(a2 + 21) )
    goto LABEL_3;
  v22 = (_DWORD *)result;
  if ( dword_4F04C44 != -1
    || (v6 = qword_4F04C68, v27 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v27 + 6) & 6) != 0)
    || *(_BYTE *)(v27 + 4) == 12 )
  {
    if ( *(_QWORD *)(a1 + 16) || v4 && (*(_BYTE *)(v4 + 33) & 1) != 0 )
    {
      *(_BYTE *)(a2 + 21) = 1;
LABEL_3:
      sub_6F40D0(a1, a2, (__int64)v6, v7, v8, v9);
LABEL_4:
      v10 = 0;
      v11 = 0;
      v12 = sub_6F6C90(a1, (_DWORD *)a2);
      goto LABEL_5;
    }
  }
  if ( *(_BYTE *)(a2 + 17) )
  {
    if ( *(_BYTE *)(a1 + 8) )
      goto LABEL_4;
    v33 = *(_QWORD *)(a1 + 24);
    v10 = (__m128i *)(v33 + 8);
    v34 = v33 + 8;
    if ( (unsigned __int8)(*(_BYTE *)(v33 + 24) - 1) <= 1u )
      sub_6E5940(v34);
    else
      sub_6E6840(v34);
    goto LABEL_55;
  }
  if ( !*(_BYTE *)(a2 + 18) )
  {
LABEL_75:
    if ( *(_BYTE *)(a1 + 8) != 1 || !*(_BYTE *)(a2 + 20) )
      goto LABEL_76;
    goto LABEL_89;
  }
  if ( !*(_BYTE *)(a2 + 19) )
  {
    if ( !v4 && *(_DWORD *)(a2 + 24) == -1 )
    {
      sub_69D070(0x8Cu, v22);
      *(_BYTE *)(a2 + 18) = 0;
    }
    goto LABEL_75;
  }
  if ( v4 )
  {
    if ( *(_BYTE *)(a1 + 8) )
    {
      v10 = v45;
      a2 = v4;
      sub_848800(a1, v4, 0, 0xA7u, v45);
    }
    else
    {
      a2 = v4;
      v10 = (__m128i *)(*(_QWORD *)(a1 + 24) + 8LL);
      sub_843D70(v10, v4, 0, 0xA7u);
    }
    if ( (*(_BYTE *)(v4 + 34) & 8) != 0 )
    {
      if ( (unsigned int)sub_6E9880((__int64)v10) )
      {
        a2 = 1622;
        if ( sub_6E53E0(5, 0x656u, &v10[4].m128i_i32[1]) )
        {
          a2 = (__int64)v10[4].m128i_i64 + 4;
          sub_684B30(0x656u, &v10[4].m128i_i32[1]);
        }
      }
    }
    goto LABEL_55;
  }
  if ( !*(_BYTE *)(a2 + 20) )
  {
    result = sub_6E5430();
    if ( (_DWORD)result )
    {
      if ( qword_4F04C50 )
        *(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) &= ~2u;
      result = sub_6851C0(0x8Cu, v22);
    }
    *(_BYTE *)(a2 + 22) = 1;
    return result;
  }
  *(_BYTE *)(a2 + 18) = 0;
  if ( *(_BYTE *)(a1 + 8) == 1 )
  {
LABEL_89:
    if ( (unsigned int)sub_6E5430() )
      sub_6851C0(0x92Fu, v22);
    sub_6E6500(a1);
  }
LABEL_76:
  sub_6E65B0(a1);
  v43 = *(_QWORD *)(a1 + 24);
  v10 = (__m128i *)(v43 + 8);
  v35 = *(_BYTE *)(a2 + 20);
  if ( unk_4D041F8 && v35 && *(_QWORD *)a2 )
  {
    if ( (unsigned int)sub_7176C0(*(_QWORD *)a2, v45)
      || (v39 = *(_QWORD *)a2, !*(_BYTE *)(*(_QWORD *)a2 + 174LL)) && *(_WORD *)(v39 + 176) == 24995
      || *(char *)(v39 + 198) < 0 )
    {
      a2 = 0;
      sub_6F69D0(v10, 0);
      goto LABEL_79;
    }
    v35 = *(_BYTE *)(a2 + 20);
  }
  a2 = v35;
  sub_6FE880(v10, v35);
LABEL_79:
  if ( !*(_BYTE *)(v2 + 18) || *(_BYTE *)(v2 + 19) || !v4 )
    goto LABEL_55;
  v29 = *(_QWORD *)(v4 + 8);
  for ( i = *(_BYTE *)(v29 + 140); i == 12; i = *(_BYTE *)(v29 + 140) )
    v29 = *(_QWORD *)(v29 + 160);
  if ( !i )
    goto LABEL_55;
  v41 = v29;
  a2 = sub_8D6740(v29);
  v37 = sub_828CC0(v10->m128i_i64, a2);
  v29 = v41;
  v38 = v37;
  if ( v37 == 5 )
  {
    if ( (*(_BYTE *)(v4 + 34) & 4) == 0 )
    {
      if ( !(unsigned int)sub_8D3B10(v41) )
        goto LABEL_87;
      v29 = v41;
      if ( (*(_BYTE *)(v41 + 179) & 0x10) == 0 )
        goto LABEL_87;
    }
    v29 = *(_QWORD *)(v29 + 160);
    if ( !v29 )
      goto LABEL_87;
    do
    {
      a2 = *(_QWORD *)(v29 + 120);
      v42 = v29;
      v40 = sub_828CC0(v10->m128i_i64, a2);
      v29 = v42;
      if ( v38 > v40 )
      {
        if ( v40 == 3 )
          goto LABEL_55;
        v38 = v40;
      }
      v29 = *(_QWORD *)(v42 + 112);
    }
    while ( v29 );
  }
  if ( v38 != 3 )
  {
LABEL_87:
    a2 = 180;
    sub_6E5C80(v38, 0xB4u, (_DWORD *)(v43 + 76));
  }
LABEL_55:
  v11 = 1;
  v12 = sub_6F7150(v10, a2, v28, v29, v30, v31);
LABEL_5:
  v15 = *(_QWORD *)(v2 + 40);
  if ( v15 )
    *(_QWORD *)(v15 + 16) = v12;
  else
    *(_QWORD *)(v2 + 32) = v12;
  *(_QWORD *)(v2 + 40) = v12;
  if ( v4 && !*(_BYTE *)(v2 + 21) )
    *(_QWORD *)(v2 + 8) = *(_QWORD *)v4;
  result = (unsigned int)*(unsigned __int8 *)(v2 + 23) - 1;
  if ( (unsigned __int8)(*(_BYTE *)(v2 + 23) - 1) <= 1u )
  {
    v16 = 0;
    v17 = *(_DWORD *)(v2 + 52);
    if ( *(_BYTE *)(v2 + 18) )
    {
      if ( !*(_QWORD *)(v2 + 8) )
      {
        v16 = 1;
        if ( !v17 )
        {
          v32 = *(_QWORD *)a1;
          if ( *(_QWORD *)a1 && *(_BYTE *)(v32 + 8) == 3 )
          {
            v32 = sub_6BBB10((_QWORD *)a1);
            v17 = *(_DWORD *)(v2 + 52);
          }
          *(_QWORD *)(v2 + 64) = v32;
          v16 = 1;
        }
      }
    }
    v18 = *(unsigned int *)(v2 + 28);
    if ( (_DWORD)v18 == v17 )
      *(_QWORD *)(v2 + 64) = a1;
    result = *(unsigned int *)(v2 + 48);
    if ( (_DWORD)result )
    {
      result = (_DWORD)v18 == (_DWORD)result;
      v11 &= result;
    }
    else if ( !(_BYTE)v16 )
    {
      return result;
    }
    if ( v11 )
    {
      v19 = sub_6F6F40(v10, 0, v18, v16, v13, v14);
      while ( *(_BYTE *)(v19 + 24) == 1 )
      {
        v20 = *(_BYTE *)(v19 + 56);
        if ( v20 == 5 )
        {
          v19 = *(_QWORD *)(v19 + 72);
        }
        else
        {
          if ( (unsigned __int8)(v20 - 105) > 2u )
            break;
          v23 = sub_72B0F0(*(_QWORD *)(v19 + 72), 0);
          if ( !v23 )
            break;
          for ( j = *(_QWORD *)(v23 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v25 = *(_QWORD *)(j + 168);
          result = (unsigned int)*(unsigned __int8 *)(v25 + 24) - 1;
          if ( (unsigned __int8)(*(_BYTE *)(v25 + 24) - 1) <= 1u )
            return result;
          v26 = *(_DWORD *)(v25 + 28);
          if ( !v26 )
            return result;
          result = *(_QWORD *)(v19 + 72);
          v19 = *(_QWORD *)(result + 16);
          if ( v26 > 1 )
          {
            if ( !v19 )
              return result;
            LODWORD(result) = v11;
            while ( 1 )
            {
              result = (unsigned int)(result + 1);
              v19 = *(_QWORD *)(v19 + 16);
              if ( v26 == (_DWORD)result )
                break;
              if ( !v19 )
                return result;
            }
          }
          if ( !v19 )
            return result;
        }
      }
      result = sub_7175E0(v19, &v44);
      if ( (_DWORD)result )
      {
        result = v44;
        if ( (*(_BYTE *)(v44 + 168) & 7) == 0 )
        {
          v21 = *(_QWORD *)(v44 + 184);
          *(_QWORD *)(v2 + 72) = v21;
          result = *(_QWORD *)(result + 176);
          if ( *(_BYTE *)(v21 + result - 1) )
            *(_QWORD *)(v2 + 72) = 0;
        }
      }
    }
  }
  return result;
}
