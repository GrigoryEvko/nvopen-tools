// Function: sub_388E560
// Address: 0x388e560
//
__int64 __fastcall sub_388E560(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r13
  int v5; // r8d
  unsigned int v6; // r12d
  int v7; // eax
  __int64 v8; // r9
  __int64 v9; // rax
  int v10; // eax
  __int64 *v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned int v15; // eax
  unsigned int v16; // eax
  int v17; // eax
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // [rsp+0h] [rbp-D0h]
  __int64 v23; // [rsp+0h] [rbp-D0h]
  unsigned int v24; // [rsp+0h] [rbp-D0h]
  _QWORD v25[2]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v26; // [rsp+20h] [rbp-B0h]
  const char *v27; // [rsp+30h] [rbp-A0h] BYREF
  char *v28; // [rsp+38h] [rbp-98h]
  __int16 v29; // [rsp+40h] [rbp-90h]
  _QWORD *v30; // [rsp+50h] [rbp-80h] BYREF
  __int64 v31; // [rsp+58h] [rbp-78h]
  _BYTE v32[112]; // [rsp+60h] [rbp-70h] BYREF

  v4 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v6 = sub_388AF10(a1, 12, "expected '(' here");
  if ( (_BYTE)v6 )
    return v6;
  v31 = 0x800000000LL;
  v7 = *(_DWORD *)(a1 + 64);
  v30 = v32;
  if ( v7 != 13 )
  {
    if ( v7 == 384 )
      goto LABEL_22;
LABEL_4:
    if ( v7 == 390 && *(_BYTE *)(a1 + 164) )
    {
      if ( *(_DWORD *)(a1 + 160) <= 0x40u )
      {
        v8 = *(_QWORD *)(a1 + 152);
LABEL_9:
        v9 = (unsigned int)v31;
        if ( (unsigned int)v31 >= HIDWORD(v31) )
        {
          v23 = v8;
          sub_16CD150((__int64)&v30, v32, 0, 8, v5, v8);
          v9 = (unsigned int)v31;
          v8 = v23;
        }
        v30[v9] = v8;
        LODWORD(v31) = v31 + 1;
        v10 = sub_3887100(v4);
        *(_DWORD *)(a1 + 64) = v10;
        while ( v10 == 4 )
        {
          v7 = sub_3887100(v4);
          *(_DWORD *)(a1 + 64) = v7;
          if ( v7 != 384 )
            goto LABEL_4;
LABEL_22:
          v15 = sub_14E4230(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
          if ( !v15 )
          {
            v25[0] = "invalid DWARF op '";
            v25[1] = a1 + 72;
            v27 = (const char *)v25;
            v26 = 1027;
            v28 = "'";
            v29 = 770;
            goto LABEL_26;
          }
          v24 = v15;
          v17 = sub_3887100(v4);
          v20 = (unsigned int)v31;
          *(_DWORD *)(a1 + 64) = v17;
          v21 = v24;
          if ( (unsigned int)v20 >= HIDWORD(v31) )
          {
            sub_16CD150((__int64)&v30, v32, 0, 8, v18, v19);
            v20 = (unsigned int)v31;
            v21 = v24;
          }
          v30[v20] = v21;
          v10 = *(_DWORD *)(a1 + 64);
          LODWORD(v31) = v31 + 1;
        }
        goto LABEL_13;
      }
      v22 = *(_DWORD *)(a1 + 160);
      if ( v22 - (unsigned int)sub_16A57B0(a1 + 152) <= 0x40 )
      {
        v8 = **(_QWORD **)(a1 + 152);
        goto LABEL_9;
      }
      v25[0] = -1;
      v27 = "element too large, limit is ";
      v28 = (char *)v25;
      v29 = 2819;
    }
    else
    {
      v27 = "expected unsigned integer";
      v29 = 259;
    }
LABEL_26:
    v16 = sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v27);
    v13 = (unsigned __int64)v30;
    v6 = v16;
    if ( v30 == (_QWORD *)v32 )
      return v6;
    goto LABEL_18;
  }
LABEL_13:
  v6 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v6 )
  {
    v11 = *(__int64 **)a1;
    if ( a3 )
      v12 = sub_15C4420(v11, v30, (unsigned int)v31, 1u, 1);
    else
      v12 = sub_15C4420(v11, v30, (unsigned int)v31, 0, 1);
    *a2 = v12;
  }
  v13 = (unsigned __int64)v30;
  if ( v30 != (_QWORD *)v32 )
LABEL_18:
    _libc_free(v13);
  return v6;
}
