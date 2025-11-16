// Function: sub_111FA90
// Address: 0x111fa90
//
__int64 __fastcall sub_111FA90(__int64 **a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  int v4; // eax
  __int64 v5; // rax
  unsigned int v6; // r12d
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rsi
  unsigned int v11; // r12d
  bool v12; // al
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rdx
  _BYTE *v25; // rax
  unsigned int v26; // r12d
  char v27; // al
  __int64 v28; // rsi
  bool v29; // r12
  unsigned int v30; // r15d
  _BYTE *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  char v34; // al
  __int64 v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v38; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v39; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v40; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v41; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v42; // [rsp+38h] [rbp-38h]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 )
    return 0;
  v4 = sub_B53900(a2);
  if ( v4 == 36 )
  {
    if ( *(_BYTE *)v2 == 42
      && (v16 = *(_QWORD *)(v2 - 64)) != 0
      && (v17 = *(_QWORD *)(v2 - 32)) != 0
      && (v3 == v16 || v3 == v17) )
    {
      **a1 = v16;
      *a1[1] = v17;
      if ( *(_BYTE *)v2 <= 0x1Cu )
        return 0;
    }
    else
    {
      v41 = 0;
      v42 = &v37;
      v8 = *(_QWORD *)(v2 + 16);
      if ( !v8 || *(_QWORD *)(v8 + 8) || *(_BYTE *)v2 != 59 )
        return 0;
      v9 = sub_995B10(&v41, *(_QWORD *)(v2 - 64));
      v10 = *(_QWORD *)(v2 - 32);
      if ( v9 && v10 )
      {
        *v42 = v10;
      }
      else
      {
        if ( !(unsigned __int8)sub_995B10(&v41, v10) )
          return 0;
        v32 = *(_QWORD *)(v2 - 64);
        if ( !v32 )
          return 0;
        *v42 = v32;
      }
      if ( !v37 )
        return 0;
      **a1 = v37;
      *a1[1] = v3;
      if ( *(_BYTE *)v2 <= 0x1Cu )
        return 0;
    }
    v6 = 1;
    *a1[2] = v2;
    return v6;
  }
  if ( v4 == 34 )
  {
    if ( *(_BYTE *)v3 == 42 )
    {
      v18 = *(_QWORD *)(v3 - 64);
      if ( v18 )
      {
        v19 = *(_QWORD *)(v3 - 32);
        if ( v19 )
        {
          if ( v2 == v18 || v2 == v19 )
          {
            **a1 = v18;
            *a1[1] = v19;
            if ( *(_BYTE *)v3 > 0x1Cu )
            {
LABEL_44:
              v6 = 1;
              *a1[2] = v3;
              return v6;
            }
            return 0;
          }
        }
      }
    }
    v41 = 0;
    v42 = &v37;
    v5 = *(_QWORD *)(v3 + 16);
    if ( v5 && !*(_QWORD *)(v5 + 8) && *(_BYTE *)v3 == 59 )
    {
      v27 = sub_995B10(&v41, *(_QWORD *)(v3 - 64));
      v28 = *(_QWORD *)(v3 - 32);
      if ( v27 && v28 )
      {
        *v42 = v28;
LABEL_62:
        if ( v37 )
        {
          **a1 = v37;
          *a1[1] = v2;
          if ( *(_BYTE *)v3 > 0x1Cu )
            goto LABEL_44;
        }
        return 0;
      }
      if ( (unsigned __int8)sub_995B10(&v41, v28) )
      {
        v33 = *(_QWORD *)(v3 - 64);
        if ( v33 )
        {
          *v42 = v33;
          goto LABEL_62;
        }
      }
    }
  }
  else
  {
    v41 = 0;
    v42 = &v37;
    if ( v4 == 32 )
    {
      if ( *(_BYTE *)v2 == 42 )
      {
        v20 = *(_QWORD *)(v2 - 64);
        if ( v20 )
        {
          v35 = *(_QWORD *)(v2 - 32);
          if ( v35 )
          {
            v38 = 0;
            v6 = sub_10081F0(&v38, v3);
            if ( (_BYTE)v6 )
            {
              v39 = 0;
              v21 = sub_993A50(&v39, v20);
              v22 = v35;
              if ( v21 || (v40 = 0, v34 = sub_993A50(&v40, v35), v22 = v35, v34) )
              {
                **a1 = v20;
                *a1[1] = v22;
                if ( *(_BYTE *)v2 > 0x1Cu )
                {
                  *a1[2] = v2;
                  return v6;
                }
                return 0;
              }
            }
          }
        }
      }
      if ( *(_BYTE *)v2 == 17 )
      {
        v11 = *(_DWORD *)(v2 + 32);
        if ( v11 <= 0x40 )
          v12 = *(_QWORD *)(v2 + 24) == 0;
        else
          v12 = v11 == (unsigned int)sub_C444A0(v2 + 24);
        if ( !v12 )
          return 0;
      }
      else
      {
        v23 = *(_QWORD *)(v2 + 8);
        v24 = (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17;
        if ( (unsigned int)v24 > 1 || *(_BYTE *)v2 > 0x15u )
          return 0;
        v25 = sub_AD7630(v2, 0, v24);
        if ( !v25 || *v25 != 17 )
        {
          if ( *(_BYTE *)(v23 + 8) == 17 )
          {
            v36 = *(_DWORD *)(v23 + 32);
            if ( v36 )
            {
              v29 = 0;
              v30 = 0;
              while ( 1 )
              {
                v31 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v2, v30);
                if ( !v31 )
                  break;
                if ( *v31 != 13 )
                {
                  if ( *v31 != 17 )
                    break;
                  v29 = sub_9867B0((__int64)(v31 + 24));
                  if ( !v29 )
                    break;
                }
                if ( v36 == ++v30 )
                {
                  if ( v29 )
                    goto LABEL_27;
                  return 0;
                }
              }
            }
          }
          return 0;
        }
        v26 = *((_DWORD *)v25 + 8);
        if ( v26 <= 0x40 )
        {
          if ( *((_QWORD *)v25 + 3) )
            return 0;
        }
        else if ( v26 != (unsigned int)sub_C444A0((__int64)(v25 + 24)) )
        {
          return 0;
        }
      }
LABEL_27:
      if ( *(_BYTE *)v3 == 42 )
      {
        v13 = *(_QWORD *)(v3 - 64);
        if ( v13 )
        {
          v14 = *(_QWORD *)(v3 - 32);
          if ( v14 )
          {
            v15 = *(_QWORD *)(v3 - 64);
            v39 = 0;
            if ( (unsigned __int8)sub_993A50(&v39, v15) || (v40 = 0, (unsigned __int8)sub_993A50(&v40, v14)) )
            {
              **a1 = v13;
              *a1[1] = v14;
              if ( *(_BYTE *)v3 > 0x1Cu )
                goto LABEL_44;
            }
          }
        }
      }
    }
  }
  return 0;
}
