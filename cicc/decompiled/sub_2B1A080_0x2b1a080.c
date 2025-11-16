// Function: sub_2B1A080
// Address: 0x2b1a080
//
__int64 *__fastcall sub_2B1A080(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 *v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v9; // r15
  __int64 *v10; // r13
  __int64 v11; // r15
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax

  v4 = a1;
  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 >> 5 > 0 )
  {
    while ( 1 )
    {
      v7 = *v4;
      if ( *(_BYTE *)*v4 == 63 )
      {
        v13 = *a3;
        if ( !*a3 )
        {
          v13 = *(_QWORD *)(v7 + 40);
          *a3 = v13;
        }
        if ( *(_QWORD *)(v7 + 40) != v13 || (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 2 )
          return v4;
      }
      else if ( !sub_2B16010(*v4) || !(unsigned __int8)sub_2B099C0(v7) )
      {
        return v4;
      }
      v9 = v4[1];
      v10 = v4 + 1;
      if ( *(_BYTE *)v9 == 63 )
      {
        v14 = *a3;
        if ( !*a3 )
        {
          v14 = *(_QWORD *)(v9 + 40);
          *a3 = v14;
        }
        if ( *(_QWORD *)(v9 + 40) != v14 || (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) != 2 )
          return v10;
        v11 = v4[2];
        v10 = v4 + 2;
        if ( *(_BYTE *)v11 != 63 )
        {
LABEL_11:
          if ( !sub_2B16010(v11) || !(unsigned __int8)sub_2B099C0(v11) )
            return v10;
          v12 = v4[3];
          v10 = v4 + 3;
          if ( *(_BYTE *)v12 == 63 )
            goto LABEL_37;
          goto LABEL_14;
        }
      }
      else
      {
        if ( !sub_2B16010(v4[1]) || !(unsigned __int8)sub_2B099C0(v9) )
          return v10;
        v11 = v4[2];
        v10 = v4 + 2;
        if ( *(_BYTE *)v11 != 63 )
          goto LABEL_11;
      }
      v15 = *a3;
      if ( !*a3 )
      {
        v15 = *(_QWORD *)(v11 + 40);
        *a3 = v15;
      }
      if ( *(_QWORD *)(v11 + 40) != v15 || (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 2 )
        return v10;
      v12 = v4[3];
      v10 = v4 + 3;
      if ( *(_BYTE *)v12 == 63 )
      {
LABEL_37:
        v16 = *a3;
        if ( !*a3 )
        {
          v16 = *(_QWORD *)(v12 + 40);
          *a3 = v16;
        }
        if ( *(_QWORD *)(v12 + 40) != v16 || (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) != 2 )
          return v10;
        goto LABEL_16;
      }
LABEL_14:
      if ( !sub_2B16010(v12) || !(unsigned __int8)sub_2B099C0(v12) )
        return v10;
LABEL_16:
      v4 += 4;
      if ( v4 == &a1[4 * (v5 >> 5)] )
      {
        v6 = (a2 - (__int64)v4) >> 3;
        break;
      }
    }
  }
  if ( v6 == 2 )
  {
LABEL_45:
    v18 = *v4;
    if ( *(_BYTE *)*v4 == 63 )
    {
      v22 = *a3;
      if ( !*a3 )
      {
        v22 = *(_QWORD *)(v18 + 40);
        *a3 = v22;
      }
      if ( *(_QWORD *)(v18 + 40) != v22 || (*(_DWORD *)(v18 + 4) & 0x7FFFFFF) != 2 )
        return v4;
    }
    else if ( !(unsigned __int8)sub_2B14730(v18) )
    {
      return v4;
    }
    ++v4;
    goto LABEL_48;
  }
  if ( v6 == 3 )
  {
    v17 = *v4;
    if ( *(_BYTE *)*v4 == 63 )
    {
      v20 = *a3;
      if ( !*a3 )
      {
        v20 = *(_QWORD *)(v17 + 40);
        *a3 = v20;
      }
      if ( v20 != *(_QWORD *)(v17 + 40) || (*(_DWORD *)(v17 + 4) & 0x7FFFFFF) != 2 )
        return v4;
    }
    else if ( !(unsigned __int8)sub_2B14730(v17) )
    {
      return v4;
    }
    ++v4;
    goto LABEL_45;
  }
  if ( v6 != 1 )
    return (__int64 *)a2;
LABEL_48:
  v19 = *v4;
  if ( *(_BYTE *)*v4 != 63 )
  {
    if ( !(unsigned __int8)sub_2B14730(v19) )
      return v4;
    return (__int64 *)a2;
  }
  v21 = *a3;
  if ( !*a3 )
  {
    v21 = *(_QWORD *)(v19 + 40);
    *a3 = v21;
  }
  if ( v21 == *(_QWORD *)(v19 + 40) && (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) == 2 )
    return (__int64 *)a2;
  return v4;
}
