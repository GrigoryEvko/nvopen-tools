// Function: sub_2A8D0D0
// Address: 0x2a8d0d0
//
void __fastcall sub_2A8D0D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  const void **v4; // r12
  unsigned __int64 v5; // r14
  const void **v6; // rbx
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r15
  __int64 v13; // r14
  unsigned int v14; // r14d
  bool v15; // al
  const void **v16; // rsi
  int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned int v20; // eax
  int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rbx
  unsigned __int64 v26; // r14
  unsigned int v27; // ecx
  __int64 v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned int v30; // edx
  __int64 v31; // rcx
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  unsigned __int64 v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-60h]
  __int64 v40; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-48h]
  unsigned int v42; // [rsp+40h] [rbp-40h]

  v3 = a2 - a1;
  v37 = a2;
  v36 = a3;
  if ( a2 - a1 <= 384 )
    return;
  if ( !a3 )
  {
    v38 = a2;
    goto LABEL_46;
  }
  v4 = (const void **)(a1 + 8);
  v35 = a1 + 32;
  while ( 2 )
  {
    --v36;
    v5 = a1
       + 8
       * (((__int64)(0xAAAAAAAAAAAAAAABLL * (v3 >> 3)) >> 1)
        + ((0xAAAAAAAAAAAAAAABLL * (v3 >> 3)) & 0xFFFFFFFFFFFFFFFELL));
    v6 = (const void **)(v37 - 16);
    if ( *(_DWORD *)(a1 + 40) <= 0x40u )
    {
      if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(v5 + 8) )
      {
        v39 = *(_QWORD *)(a1 + 32);
        if ( (int)sub_C4C880(v35, v5 + 8) < 0 )
          goto LABEL_33;
        if ( v39 == *(_QWORD *)(v37 - 16) )
          goto LABEL_40;
        goto LABEL_8;
      }
    }
    else if ( !sub_C43C50(v35, (const void **)(v5 + 8)) )
    {
      if ( (int)sub_C4C880(v35, v5 + 8) >= 0 )
        goto LABEL_7;
LABEL_33:
      if ( *(_DWORD *)(v5 + 16) <= 0x40u )
      {
        if ( *(_QWORD *)(v5 + 8) != *(_QWORD *)(v37 - 16) )
        {
LABEL_35:
          if ( (int)sub_C4C880(v5 + 8, (__int64)v6) < 0 )
          {
LABEL_36:
            v21 = *(_DWORD *)(a1 + 16);
            v22 = *(_QWORD *)a1;
            *(_DWORD *)(a1 + 16) = 0;
            v23 = *(_QWORD *)(a1 + 8);
            *(_QWORD *)a1 = *(_QWORD *)v5;
            *(_QWORD *)(a1 + 8) = *(_QWORD *)(v5 + 8);
            *(_DWORD *)(a1 + 16) = *(_DWORD *)(v5 + 16);
            *(_QWORD *)v5 = v22;
            *(_QWORD *)(v5 + 8) = v23;
            *(_DWORD *)(v5 + 16) = v21;
            v8 = *(_DWORD *)(a1 + 40);
            goto LABEL_10;
          }
LABEL_59:
          if ( *(_DWORD *)(a1 + 40) <= 0x40u )
          {
            if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(v37 - 16) )
            {
LABEL_61:
              if ( (int)sub_C4C880(v35, (__int64)v6) < 0 )
                goto LABEL_70;
              goto LABEL_62;
            }
          }
          else if ( !sub_C43C50(v35, v6) )
          {
            goto LABEL_61;
          }
          if ( sub_B445A0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v37 - 24)) )
          {
LABEL_70:
            sub_2A8AEC0((__int64 *)a1, (__int64 *)(v37 - 24));
            v8 = *(_DWORD *)(a1 + 40);
            goto LABEL_10;
          }
LABEL_62:
          sub_2A8AEC0((__int64 *)a1, (__int64 *)(a1 + 24));
          v8 = *(_DWORD *)(a1 + 40);
          goto LABEL_10;
        }
      }
      else if ( !sub_C43C50(v5 + 8, v6) )
      {
        goto LABEL_35;
      }
      if ( sub_B445A0(*(_QWORD *)v5, *(_QWORD *)(v37 - 24)) )
        goto LABEL_36;
      goto LABEL_59;
    }
    if ( sub_B445A0(*(_QWORD *)(a1 + 24), *(_QWORD *)v5) )
      goto LABEL_33;
    if ( *(_DWORD *)(a1 + 40) > 0x40u )
    {
LABEL_7:
      if ( !sub_C43C50(v35, v6) )
        goto LABEL_8;
LABEL_40:
      if ( sub_B445A0(*(_QWORD *)(a1 + 24), *(_QWORD *)(v37 - 24)) )
        goto LABEL_9;
LABEL_41:
      if ( *(_DWORD *)(v5 + 16) <= 0x40u )
      {
        if ( *(_QWORD *)(v5 + 8) != *(_QWORD *)(v37 - 16) )
        {
LABEL_43:
          if ( (int)sub_C4C880(v5 + 8, (__int64)v6) < 0 )
            goto LABEL_70;
LABEL_67:
          sub_2A8AEC0((__int64 *)a1, (__int64 *)v5);
          v8 = *(_DWORD *)(a1 + 40);
          goto LABEL_10;
        }
      }
      else if ( !sub_C43C50(v5 + 8, v6) )
      {
        goto LABEL_43;
      }
      if ( sub_B445A0(*(_QWORD *)v5, *(_QWORD *)(v37 - 24)) )
        goto LABEL_70;
      goto LABEL_67;
    }
    if ( *(_QWORD *)(a1 + 32) == *(_QWORD *)(v37 - 16) )
      goto LABEL_40;
LABEL_8:
    if ( (int)sub_C4C880(v35, (__int64)v6) >= 0 )
      goto LABEL_41;
LABEL_9:
    v7 = *(_QWORD *)a1;
    v8 = *(_DWORD *)(a1 + 16);
    v9 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)a1 = *(_QWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 8) = v10;
    LODWORD(v10) = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v9;
    *(_DWORD *)(a1 + 16) = v10;
    *(_DWORD *)(a1 + 40) = v8;
LABEL_10:
    v11 = a1 + 24;
    v12 = v37;
    v38 = a1 + 24;
    v13 = a1 + 32;
    if ( v8 <= 0x40 )
      goto LABEL_25;
LABEL_11:
    if ( sub_C43C50(v13, v4) )
    {
      while ( sub_B445A0(*(_QWORD *)v11, *(_QWORD *)a1) )
      {
        v20 = *(_DWORD *)(v11 + 40);
        v11 += 24LL;
LABEL_24:
        v38 = v11;
        v13 = v11 + 8;
        if ( v20 > 0x40 )
          goto LABEL_11;
LABEL_25:
        if ( *(_QWORD *)(v11 + 8) != *(_QWORD *)(a1 + 8) )
          goto LABEL_12;
      }
    }
    else
    {
LABEL_12:
      if ( (int)sub_C4C880(v13, (__int64)v4) < 0 )
      {
LABEL_23:
        v20 = *(_DWORD *)(v11 + 40);
        v11 += 24LL;
        goto LABEL_24;
      }
    }
    v14 = *(_DWORD *)(a1 + 16);
    for ( v12 -= 24LL; ; v12 -= 24LL )
    {
      v16 = (const void **)(v12 + 8);
      if ( v14 <= 0x40 )
        break;
      v15 = sub_C43C50((__int64)v4, v16);
      v16 = (const void **)(v12 + 8);
      if ( v15 )
        goto LABEL_19;
LABEL_15:
      if ( (int)sub_C4C880((__int64)v4, (__int64)v16) >= 0 )
        goto LABEL_21;
LABEL_16:
      ;
    }
    if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(v12 + 8) )
      goto LABEL_15;
LABEL_19:
    if ( sub_B445A0(*(_QWORD *)a1, *(_QWORD *)v12) )
    {
      v14 = *(_DWORD *)(a1 + 16);
      goto LABEL_16;
    }
LABEL_21:
    if ( v11 < v12 )
    {
      v17 = *(_DWORD *)(v11 + 16);
      v18 = *(_QWORD *)v11;
      *(_DWORD *)(v11 + 16) = 0;
      v19 = *(_QWORD *)(v11 + 8);
      *(_QWORD *)v11 = *(_QWORD *)v12;
      *(_QWORD *)(v11 + 8) = *(_QWORD *)(v12 + 8);
      *(_DWORD *)(v11 + 16) = *(_DWORD *)(v12 + 16);
      *(_QWORD *)v12 = v18;
      *(_QWORD *)(v12 + 8) = v19;
      *(_DWORD *)(v12 + 16) = v17;
      goto LABEL_23;
    }
    sub_2A8D0D0(v11, v37, v36);
    v3 = v11 - a1;
    if ( (__int64)(v11 - a1) > 384 )
    {
      if ( v36 )
      {
        v37 = v11;
        continue;
      }
LABEL_46:
      v24 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
      v25 = (v24 - 2) >> 1;
      v26 = a1 + 8 * (v25 + ((v24 - 2) & 0xFFFFFFFFFFFFFFFELL));
      while ( 1 )
      {
        v27 = *(_DWORD *)(v26 + 16);
        v28 = *(_QWORD *)v26;
        *(_DWORD *)(v26 + 16) = 0;
        v29 = *(_QWORD *)(v26 + 8);
        v40 = v28;
        v42 = v27;
        v41 = v29;
        sub_2A8B5A0(a1, v25, v24, (__int64)&v40);
        if ( v42 > 0x40 && v41 )
          j_j___libc_free_0_0(v41);
        v26 -= 24LL;
        if ( !v25 )
          break;
        --v25;
      }
      do
      {
        v38 -= 24;
        v30 = *(_DWORD *)(v38 + 16);
        v31 = *(_QWORD *)v38;
        *(_DWORD *)(v38 + 16) = 0;
        v32 = *(_QWORD *)a1;
        v33 = *(_QWORD *)(v38 + 8);
        v42 = v30;
        *(_QWORD *)v38 = v32;
        v34 = *(_QWORD *)(a1 + 8);
        v40 = v31;
        *(_QWORD *)(v38 + 8) = v34;
        LODWORD(v34) = *(_DWORD *)(a1 + 16);
        v41 = v33;
        *(_DWORD *)(v38 + 16) = v34;
        *(_DWORD *)(a1 + 16) = 0;
        sub_2A8B5A0(a1, 0, 0xAAAAAAAAAAAAAAABLL * ((v38 - a1) >> 3), (__int64)&v40);
        if ( v42 > 0x40 )
        {
          if ( v41 )
            j_j___libc_free_0_0(v41);
        }
      }
      while ( v38 - a1 > 24 );
    }
    break;
  }
}
