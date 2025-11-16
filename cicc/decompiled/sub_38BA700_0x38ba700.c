// Function: sub_38BA700
// Address: 0x38ba700
//
char __fastcall sub_38BA700(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  char v12; // r13
  char v13; // al
  char *v14; // rsi
  size_t v15; // r13
  char *v16; // r9
  char *v17; // rax
  char *v18; // rax
  char *v19; // rdi
  char *v21; // [rsp-C8h] [rbp-C8h]
  size_t v22; // [rsp-B0h] [rbp-B0h] BYREF
  char *v23; // [rsp-A8h] [rbp-A8h] BYREF
  size_t v24; // [rsp-A0h] [rbp-A0h]
  _BYTE v25[16]; // [rsp-98h] [rbp-98h] BYREF
  char *v26; // [rsp-88h] [rbp-88h] BYREF
  size_t v27; // [rsp-80h] [rbp-80h]
  _QWORD v28[2]; // [rsp-78h] [rbp-78h] BYREF
  void *v29; // [rsp-68h] [rbp-68h] BYREF
  __int64 v30; // [rsp-60h] [rbp-60h]
  __int64 v31; // [rsp-58h] [rbp-58h]
  __int64 v32; // [rsp-50h] [rbp-50h]
  int v33; // [rsp-48h] [rbp-48h]
  char **v34; // [rsp-40h] [rbp-40h]

  LOBYTE(v4) = *(_BYTE *)(a2 + 33) & 3;
  if ( (_BYTE)v4 == 2 )
  {
    LOBYTE(v4) = sub_15E4F60(a2);
    if ( !(_BYTE)v4 )
    {
      v8 = *(_QWORD *)(a1 + 24);
      v9 = *(_QWORD *)(a1 + 16) - v8;
      if ( *(_DWORD *)(a3 + 44) == 15 && *(_DWORD *)(a3 + 48) == 14 )
      {
        if ( v9 <= 8 )
        {
          sub_16E7EE0(a1, " /EXPORT:", 9u);
        }
        else
        {
          *(_BYTE *)(v8 + 8) = 58;
          *(_QWORD *)v8 = 0x54524F5058452F20LL;
          *(_QWORD *)(a1 + 24) += 9LL;
        }
      }
      else if ( v9 <= 8 )
      {
        sub_16E7EE0(a1, " -export:", 9u);
      }
      else
      {
        *(_BYTE *)(v8 + 8) = 58;
        *(_QWORD *)v8 = 0x74726F7078652D20LL;
        *(_QWORD *)(a1 + 24) += 9LL;
      }
      if ( *(_DWORD *)(a3 + 44) != 15 || (v11 = *(_DWORD *)(a3 + 48), v11 != 1) && v11 != 16 )
      {
        sub_38B9BB0(a4, a1, a2);
        goto LABEL_8;
      }
      v34 = &v23;
      v23 = v25;
      v29 = &unk_49EFBE0;
      v24 = 0;
      v25[0] = 0;
      v33 = 1;
      v32 = 0;
      v31 = 0;
      v30 = 0;
      sub_38B9BB0(a4, (__int64)&v29, a2);
      if ( v32 != v30 )
        sub_16E7BA0((__int64 *)&v29);
      v12 = *v23;
      switch ( *(_DWORD *)(sub_1632FA0(*(_QWORD *)(a2 + 40)) + 16) )
      {
        case 0:
        case 1:
        case 3:
        case 5:
          v13 = 0;
          break;
        case 2:
        case 4:
          v13 = 95;
          break;
      }
      if ( v12 != v13 )
      {
        sub_16E7EE0(a1, v23, v24);
LABEL_25:
        sub_16E7BC0((__int64 *)&v29);
        if ( v23 != v25 )
          j_j___libc_free_0((unsigned __int64)v23);
LABEL_8:
        v4 = *(_QWORD *)(a2 + 24);
        if ( *(_BYTE *)(v4 + 8) == 12 )
          return v4;
        v10 = *(_QWORD *)(a1 + 24);
        v4 = *(_QWORD *)(a1 + 16) - v10;
        if ( *(_DWORD *)(a3 + 44) == 15 && *(_DWORD *)(a3 + 48) == 14 )
        {
          if ( v4 > 4 )
          {
            *(_DWORD *)v10 = 1413563436;
            *(_BYTE *)(v10 + 4) = 65;
            *(_QWORD *)(a1 + 24) += 5LL;
            return v4;
          }
          v14 = ",DATA";
        }
        else
        {
          if ( v4 > 4 )
          {
            *(_DWORD *)v10 = 1952539692;
            *(_BYTE *)(v10 + 4) = 97;
            *(_QWORD *)(a1 + 24) += 5LL;
            return v4;
          }
          v14 = ",data";
        }
        LOBYTE(v4) = sub_16E7EE0(a1, v14, 5u);
        return v4;
      }
      if ( !v24 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::substr", 1u, 0);
      v15 = v24 - 1;
      v16 = v23;
      v26 = (char *)v28;
      v22 = v24 - 1;
      if ( v24 - 1 > 0xF )
      {
        v21 = v23;
        v18 = (char *)sub_22409D0((__int64)&v26, &v22, 0);
        v16 = v21;
        v26 = v18;
        v19 = v18;
        v28[0] = v22;
      }
      else
      {
        if ( v24 == 2 )
        {
          LOBYTE(v28[0]) = v23[1];
          v17 = (char *)v28;
          goto LABEL_37;
        }
        if ( v24 == 1 )
        {
          v17 = (char *)v28;
          goto LABEL_37;
        }
        v19 = (char *)v28;
      }
      memcpy(v19, v16 + 1, v15);
      v15 = v22;
      v17 = v26;
LABEL_37:
      v27 = v15;
      v17[v15] = 0;
      sub_16E7EE0(a1, v26, v27);
      if ( v26 != (char *)v28 )
        j_j___libc_free_0((unsigned __int64)v26);
      goto LABEL_25;
    }
  }
  return v4;
}
