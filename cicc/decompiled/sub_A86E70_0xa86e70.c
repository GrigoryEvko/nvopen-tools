// Function: sub_A86E70
// Address: 0xa86e70
//
unsigned __int64 __fastcall sub_A86E70(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rsi
  _QWORD *v6; // r15
  _QWORD *v7; // r14
  _QWORD *v8; // rdi
  unsigned __int64 result; // rax
  const char *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // r14
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+28h] [rbp-78h] BYREF
  __int64 v37[4]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v38; // [rsp+50h] [rbp-50h]

  v32 = a1 + 72;
  if ( !(unsigned __int8)sub_B2FC80(a1) && !(unsigned __int8)sub_B2D610(a1, 72) )
  {
    v31 = *(_QWORD *)(a1 + 80);
    while ( v31 != v32 )
    {
      v11 = v31;
      v12 = *(_QWORD *)(v31 + 32);
      v31 = *(_QWORD *)(v31 + 8);
      v34 = v11 + 24;
      if ( v11 + 24 != v12 )
      {
        while ( 2 )
        {
          v13 = v12;
          v12 = *(_QWORD *)(v12 + 8);
          switch ( *(_BYTE *)(v13 - 24) )
          {
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x21:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x37:
            case 0x38:
            case 0x39:
            case 0x3A:
            case 0x3B:
            case 0x3C:
            case 0x3D:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x56:
            case 0x57:
            case 0x58:
            case 0x59:
            case 0x5A:
            case 0x5B:
            case 0x5C:
            case 0x5D:
            case 0x5E:
            case 0x5F:
            case 0x60:
              goto LABEL_28;
            case 0x22:
            case 0x28:
              if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v13 + 48), 72) || (unsigned __int8)sub_B49560(v13 - 24, 72) )
              {
                if ( *(_BYTE *)(v13 - 24) != 85 )
                  goto LABEL_27;
                v29 = *(_QWORD *)(v13 - 56);
                if ( !v29 )
                  goto LABEL_27;
                if ( *(_BYTE *)v29 )
                  goto LABEL_27;
                v16 = *(_QWORD *)(v13 + 56);
                if ( *(_QWORD *)(v29 + 24) != v16
                  || (*(_BYTE *)(v29 + 33) & 0x20) == 0
                  || !(unsigned __int8)sub_B5A1B0(v13 - 24) )
                {
                  goto LABEL_27;
                }
              }
              goto LABEL_28;
            case 0x55:
              if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v13 + 48), 72) || (unsigned __int8)sub_B49560(v13 - 24, 72) )
              {
                if ( *(_BYTE *)(v13 - 24) != 85
                  || (v30 = *(_QWORD *)(v13 - 56)) == 0
                  || *(_BYTE *)v30
                  || (v16 = *(_QWORD *)(v13 + 56), *(_QWORD *)(v30 + 24) != v16)
                  || (*(_BYTE *)(v30 + 33) & 0x20) == 0
                  || !(unsigned __int8)sub_B5A1B0(v13 - 24) )
                {
LABEL_27:
                  v17 = (__int64 *)sub_BD5C60(v13 - 24, 72, v16);
                  *(_QWORD *)(v13 + 48) = sub_A7B980((__int64 *)(v13 + 48), v17, -1, 72);
                  v19 = (__int64 *)sub_BD5C60(v13 - 24, v17, v18);
                  *(_QWORD *)(v13 + 48) = sub_A7A090((__int64 *)(v13 + 48), v19, -1, 23);
                }
              }
LABEL_28:
              if ( v34 == v12 )
                break;
              continue;
            default:
              goto LABEL_59;
          }
          break;
        }
      }
    }
  }
  v36 = *(_QWORD *)(a1 + 120);
  v2 = sub_A74610(&v36);
  sub_A751C0((__int64)v37, **(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL), v2, 3);
  sub_B2D550(a1, v37);
  sub_A7BDC0(v38, (__int64)v37);
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1);
    v3 = *(_QWORD *)(a1 + 96);
    v33 = v3 + 40LL * *(_QWORD *)(a1 + 104);
    if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a1);
      v3 = *(_QWORD *)(a1 + 96);
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 96);
    v33 = v3 + 40LL * *(_QWORD *)(a1 + 104);
  }
  for ( ; v33 != v3; v3 += 40 )
  {
    v4 = sub_B2BDE0(v3);
    sub_A751C0((__int64)v37, *(_QWORD *)(v3 + 8), v4, 3);
    v5 = (__int64)v37;
    sub_B2BE60(v3, v37);
    v6 = v38;
    while ( v6 )
    {
      v7 = v6;
      sub_A7BDC0((_QWORD *)v6[3], v5);
      v8 = (_QWORD *)v6[4];
      v6 = (_QWORD *)v6[2];
      if ( v8 != v7 + 7 )
        _libc_free(v8, v5);
      v5 = 88;
      j_j___libc_free_0(v7, 88);
    }
  }
  v37[0] = sub_B2D7E0(a1, "implicit-section-name", 21);
  if ( v37[0] && sub_A71840((__int64)v37) )
  {
    v14 = sub_A72240(v37);
    sub_B31A00(a1, v14, v15);
    sub_B2D4A0(a1, "implicit-section-name", 21);
  }
  result = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v32 != result )
  {
    v10 = "amdgpu-unsafe-fp-atomics";
    result = sub_B2D7E0(a1, "amdgpu-unsafe-fp-atomics", 24);
    v37[0] = result;
    if ( result )
    {
      if ( (unsigned __int8)sub_A72A30(v37) )
      {
        v20 = *(_QWORD *)(a1 + 80);
        while ( v32 != v20 )
        {
          v21 = v20;
          v20 = *(_QWORD *)(v20 + 8);
          v22 = *(_QWORD *)(v21 + 32);
          v23 = v21 + 24;
          while ( v23 != v22 )
          {
            v24 = v22;
            v22 = *(_QWORD *)(v22 + 8);
            v25 = *(unsigned __int8 *)(v24 - 24);
            if ( v25 == 66 )
            {
              if ( ((*(_WORD *)(v24 - 22) >> 4) & 0x1Fu) - 11 <= 3 )
              {
                v26 = v24 - 24;
                v35 = v20;
                v27 = sub_BD5C60(v24 - 24, v10, v24);
                v28 = sub_B9C770(v27, 0, 0, 0, 1);
                sub_B9A090(v26, "amdgpu.no.fine.grained.host.memory", 34, v28);
                sub_B9A090(v26, "amdgpu.no.remote.memory.access", 30, v28);
                v10 = "amdgpu.ignore.denormal.mode";
                sub_B9A090(v26, "amdgpu.ignore.denormal.mode", 27, v28);
                v20 = v35;
              }
            }
            else if ( (unsigned int)(v25 - 29) <= 0x25 )
            {
              if ( (unsigned int)(v25 - 30) > 0x23 )
                goto LABEL_59;
            }
            else if ( (unsigned int)(v25 - 67) > 0x1D )
            {
LABEL_59:
              BUG();
            }
          }
        }
      }
      return sub_B2D4A0(a1, "amdgpu-unsafe-fp-atomics", 24);
    }
  }
  return result;
}
