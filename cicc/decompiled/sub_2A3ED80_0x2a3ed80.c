// Function: sub_2A3ED80
// Address: 0x2a3ed80
//
void __fastcall sub_2A3ED80(__int64 **a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  __int64 *v6; // r15
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // r15d
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rcx
  int v19; // r8d
  _BYTE *v20; // rsi
  int v21; // ecx
  unsigned int v22; // eax
  unsigned int v23; // r8d
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 *v29; // [rsp+18h] [rbp-78h] BYREF
  unsigned int *v30; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v31; // [rsp+28h] [rbp-68h]
  _BYTE v32[16]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v33; // [rsp+40h] [rbp-50h] BYREF
  size_t v34; // [rsp+48h] [rbp-48h]
  _QWORD v35[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( sub_BA91D0((__int64)a1, "kcfi", 4u) )
  {
    v6 = *a1;
    v29 = *a1;
    if ( a3 )
    {
      v30 = (unsigned int *)v32;
      sub_2A3E750((__int64 *)&v30, a3, (__int64)&a3[a4]);
    }
    else
    {
      v31 = 0;
      v30 = (unsigned int *)v32;
      v32[0] = 0;
    }
    if ( sub_BA91D0((__int64)a1, "cfi-normalize-integers", 0x16u) )
    {
      if ( 0x3FFFFFFFFFFFFFFFLL - v31 <= 0xA )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v30, ".normalized", 0xBu);
    }
    v7 = sub_CBF4B0(v30, v31);
    v8 = sub_BCB2D0(v6);
    v9 = sub_ACD640(v8, v7, 0);
    v33 = (_QWORD *)sub_B8C140((__int64)&v29, v9, v10, v11);
    v12 = sub_B9C770(v6, (__int64 *)&v33, (__int64 *)1, 0, 1);
    sub_B99110(a2, 36, v12);
    v13 = sub_BA91D0((__int64)a1, "kcfi-offset", 0xBu);
    if ( !v13 )
      goto LABEL_12;
    v14 = *(_QWORD *)(v13 + 136);
    if ( !v14 )
      goto LABEL_12;
    v15 = *(_QWORD *)(v14 + 24);
    if ( *(_DWORD *)(v14 + 32) > 0x40u )
      v15 = *(_QWORD *)v15;
    v16 = v15;
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    if ( (unsigned int)v15 > 9 )
    {
      if ( (unsigned int)v15 <= 0x63 )
      {
        v33 = v35;
        sub_2240A50((__int64 *)&v33, 2u, 0);
        v20 = v33;
      }
      else
      {
        if ( (unsigned int)v15 <= 0x3E7 )
        {
          v17 = 3;
        }
        else
        {
          v15 = (unsigned int)v15;
          if ( (unsigned int)v15 <= 0x270F )
          {
            v17 = 4;
          }
          else
          {
            LODWORD(v17) = 1;
            while ( 1 )
            {
              v18 = v15;
              v19 = v17;
              v17 = (unsigned int)(v17 + 4);
              v15 /= 0x2710u;
              if ( v18 <= 0x1869F )
                break;
              if ( (unsigned int)v15 <= 0x63 )
              {
                v17 = (unsigned int)(v19 + 5);
                v33 = v35;
                goto LABEL_26;
              }
              if ( (unsigned int)v15 <= 0x3E7 )
              {
                v17 = (unsigned int)(v19 + 6);
                break;
              }
              if ( (unsigned int)v15 <= 0x270F )
              {
                v17 = (unsigned int)(v19 + 7);
                break;
              }
            }
          }
        }
        v33 = v35;
LABEL_26:
        sub_2240A50((__int64 *)&v33, v17, 0);
        v20 = v33;
        v21 = v34 - 1;
        do
        {
          v22 = v16 % 0x64;
          v23 = v16;
          v16 /= 0x64u;
          v24 = 2 * v22;
          v25 = (unsigned int)(v24 + 1);
          LOBYTE(v24) = a00010203040506[v24];
          v20[v21] = a00010203040506[v25];
          v26 = (unsigned int)(v21 - 1);
          v21 -= 2;
          v20[v26] = v24;
        }
        while ( v23 > 0x270F );
        if ( v23 <= 0x3E7 )
          goto LABEL_29;
      }
      v27 = 2 * v16;
      v20[1] = a00010203040506[(unsigned int)(v27 + 1)];
      *v20 = a00010203040506[v27];
LABEL_30:
      sub_B2CD60(a2, "patchable-function-prefix", 0x19u, v33, v34);
      if ( v33 != v35 )
        j_j___libc_free_0((unsigned __int64)v33);
LABEL_12:
      if ( v30 != (unsigned int *)v32 )
        j_j___libc_free_0((unsigned __int64)v30);
      return;
    }
    v33 = v35;
    sub_2240A50((__int64 *)&v33, 1u, 0);
    v20 = v33;
LABEL_29:
    *v20 = v16 + 48;
    goto LABEL_30;
  }
}
