// Function: sub_D11900
// Address: 0xd11900
//
__int64 __fastcall sub_D11900(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rcx
  unsigned __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 *v10; // rsi
  __int64 **v11; // rbx
  __int64 **v12; // r12
  char *v13; // rdi
  unsigned __int8 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // r8
  char v17; // al
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // rdx
  char v21; // al
  __int64 v22; // rax
  __int64 v23; // r12
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-F8h]
  __int64 v27; // [rsp+10h] [rbp-F0h]
  __int64 v28; // [rsp+18h] [rbp-E8h]
  __int64 v30; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v31; // [rsp+60h] [rbp-A0h] BYREF
  char *v32; // [rsp+68h] [rbp-98h]
  int v33; // [rsp+70h] [rbp-90h]
  char v34; // [rsp+78h] [rbp-88h] BYREF
  _QWORD v35[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v36; // [rsp+90h] [rbp-70h]
  __int64 v37; // [rsp+98h] [rbp-68h]
  __int64 v38; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v39; // [rsp+A8h] [rbp-58h]
  __int64 v40; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-48h]

  v2 = a2[1];
  if ( sub_B2FC80(v2) && !(unsigned __int8)sub_B2D610(v2, 24) )
  {
    v23 = a2[3];
    v41 = 0;
    v35[0] = a1[8];
    if ( v23 == a2[4] )
    {
      sub_D10B90(a2 + 2, v23, (__int64)&v38, v35);
      v24 = v41;
    }
    else
    {
      if ( !v23 )
      {
        a2[3] = 40;
        goto LABEL_72;
      }
      *(_BYTE *)(v23 + 24) = 0;
      if ( !(_BYTE)v41 )
      {
        *(_QWORD *)(v23 + 32) = v35[0];
        a2[3] += 40LL;
LABEL_72:
        ++*(_DWORD *)(v35[0] + 40LL);
        goto LABEL_2;
      }
      *(_QWORD *)v23 = 6;
      *(_QWORD *)(v23 + 8) = 0;
      v25 = v40;
      v19 = v40 == 0;
      *(_QWORD *)(v23 + 16) = v40;
      if ( v25 != -4096 && !v19 && v25 != -8192 )
        sub_BD6050((unsigned __int64 *)v23, v38 & 0xFFFFFFFFFFFFFFF8LL);
      *(_BYTE *)(v23 + 24) = 1;
      v24 = v41;
      *(_QWORD *)(v23 + 32) = v35[0];
      a2[3] += 40LL;
    }
    if ( v24 )
    {
      LOBYTE(v41) = 0;
      if ( v40 != -4096 && v40 != 0 && v40 != -8192 )
        sub_BD60C0(&v38);
    }
    goto LABEL_72;
  }
LABEL_2:
  result = *(_QWORD *)(v2 + 80);
  v26 = v2 + 72;
  v28 = result;
  if ( result != v2 + 72 )
  {
    while ( 1 )
    {
      if ( !v28 )
        BUG();
      v5 = *(_QWORD *)(v28 + 32);
      if ( v5 != v28 + 24 )
        break;
LABEL_42:
      result = *(_QWORD *)(v28 + 8);
      v28 = result;
      if ( v26 == result )
        return result;
    }
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v5 - 24) - 34) > 0x33u )
        goto LABEL_41;
      v6 = 0x8000000000041LL;
      if ( !_bittest64(&v6, (unsigned int)*(unsigned __int8 *)(v5 - 24) - 34) )
        goto LABEL_41;
      v7 = *(_QWORD *)(v5 - 56);
      if ( v7 && !*(_BYTE *)v7 && *(_QWORD *)(v7 + 24) == *(_QWORD *)(v5 + 56) )
      {
        if ( (unsigned int)(*(_DWORD *)(v7 + 36) - 68) <= 3 )
          goto LABEL_20;
        v8 = sub_D110B0(a1, v7);
      }
      else
      {
        v8 = a1[8];
      }
      v35[0] = v8;
      v38 = 6;
      v39 = 0;
      v40 = v5 - 24;
      if ( v5 != -4072 && v5 != -8168 )
        sub_BD73F0((__int64)&v38);
      LOBYTE(v41) = 1;
      v9 = a2[3];
      if ( v9 == a2[4] )
      {
        sub_D10B90(a2 + 2, a2[3], (__int64)&v38, v35);
        v21 = v41;
      }
      else
      {
        if ( !v9 )
        {
          a2[3] = 40;
LABEL_54:
          LOBYTE(v41) = 0;
          if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
            sub_BD60C0(&v38);
          goto LABEL_19;
        }
        *(_BYTE *)(v9 + 24) = 0;
        if ( !(_BYTE)v41 )
        {
          *(_QWORD *)(v9 + 32) = v35[0];
          a2[3] += 40LL;
          goto LABEL_19;
        }
        *(_QWORD *)v9 = 6;
        *(_QWORD *)(v9 + 8) = 0;
        v22 = v40;
        v19 = v40 == -4096;
        *(_QWORD *)(v9 + 16) = v40;
        if ( v22 != 0 && !v19 && v22 != -8192 )
          sub_BD6050((unsigned __int64 *)v9, v38 & 0xFFFFFFFFFFFFFFF8LL);
        *(_BYTE *)(v9 + 24) = 1;
        v21 = v41;
        *(_QWORD *)(v9 + 32) = v35[0];
        a2[3] += 40LL;
      }
      if ( v21 )
        goto LABEL_54;
LABEL_19:
      ++*(_DWORD *)(v35[0] + 40LL);
LABEL_20:
      v10 = &v38;
      v38 = (__int64)&v40;
      v39 = 0x400000000LL;
      sub_E33A00(v5 - 24);
      v11 = (__int64 **)v38;
      v12 = (__int64 **)(v38 + 8LL * (unsigned int)v39);
      if ( (__int64 **)v38 != v12 )
      {
        while ( 1 )
        {
          v10 = *v11;
          sub_E33C60(&v31, *v11);
          if ( !v33 && !sub_B491E0(v31) )
            break;
          v13 = v32;
          v14 = *(unsigned __int8 **)(v31
                                    + 32 * (*(unsigned int *)v32 - (unsigned __int64)(*(_DWORD *)(v31 + 4) & 0x7FFFFFF)));
          if ( v14 )
            goto LABEL_23;
LABEL_30:
          if ( v13 != &v34 )
            _libc_free(v13, v10);
          if ( v12 == ++v11 )
          {
            v12 = (__int64 **)v38;
            goto LABEL_39;
          }
        }
        v14 = *(unsigned __int8 **)(v31 - 32);
        if ( v14 )
        {
LABEL_23:
          v10 = (__int64 *)sub_BD3990(v14, (__int64)v10);
          if ( !*(_BYTE *)v10 )
          {
            v15 = sub_D110B0(a1, (unsigned __int64)v10);
            v37 = 0;
            v16 = a2[3];
            v30 = v15;
            if ( v16 == a2[4] )
            {
              v10 = (__int64 *)v16;
              sub_D10B90(a2 + 2, v16, (__int64)v35, &v30);
              v17 = v37;
              goto LABEL_45;
            }
            if ( v16 )
            {
              *(_BYTE *)(v16 + 24) = 0;
              if ( !(_BYTE)v37 )
              {
                *(_QWORD *)(v16 + 32) = v15;
                a2[3] += 40LL;
                goto LABEL_28;
              }
              *(_QWORD *)v16 = 6;
              *(_QWORD *)(v16 + 8) = 0;
              v18 = v36;
              v19 = v36 == 0;
              *(_QWORD *)(v16 + 16) = v36;
              if ( v18 != -4096 && !v19 && v18 != -8192 )
              {
                v27 = v16;
                v10 = (__int64 *)(v35[0] & 0xFFFFFFFFFFFFFFF8LL);
                sub_BD6050((unsigned __int64 *)v16, v35[0] & 0xFFFFFFFFFFFFFFF8LL);
                v16 = v27;
              }
              v20 = v30;
              *(_BYTE *)(v16 + 24) = 1;
              v17 = v37;
              *(_QWORD *)(v16 + 32) = v20;
              a2[3] += 40LL;
LABEL_45:
              if ( v17 )
              {
                LOBYTE(v37) = 0;
                if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
                  sub_BD60C0(v35);
              }
            }
            else
            {
              a2[3] = 40;
            }
LABEL_28:
            ++*(_DWORD *)(v30 + 40);
          }
        }
        v13 = v32;
        goto LABEL_30;
      }
LABEL_39:
      if ( v12 != (__int64 **)&v40 )
        _libc_free(v12, v10);
LABEL_41:
      v5 = *(_QWORD *)(v5 + 8);
      if ( v28 + 24 == v5 )
        goto LABEL_42;
    }
  }
  return result;
}
