// Function: sub_ADABF0
// Address: 0xadabf0
//
__int64 __fastcall sub_ADABF0(__int64 a1, __int64 a2, __int64 a3, __int64 **a4, char a5, __int64 a6)
{
  int v10; // edi
  __int64 **v11; // r15
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 result; // rax
  __int64 v16; // rcx
  unsigned __int8 *v17; // r14
  __int64 *v18; // rdx
  unsigned __int8 v19; // r8
  _DWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int8 v23; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v24; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v25; // [rsp+8h] [rbp-A8h]
  __int64 *v26; // [rsp+8h] [rbp-A8h]
  __int64 *v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+10h] [rbp-A0h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+18h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+28h] [rbp-88h]
  __int64 v37; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-78h]
  char v39; // [rsp+40h] [rbp-70h]
  __int64 v40; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+58h] [rbp-58h]
  __int64 v42; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-48h]
  char v44; // [rsp+70h] [rbp-40h]

  if ( a4 == *(__int64 ***)(a1 + 8) )
  {
    v12 = (_QWORD *)(a2 + 8 * a3);
    v13 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    if ( (_QWORD *)a2 == v12 )
      return a1;
    v14 = (_QWORD *)a2;
    while ( *v13 == *v14 )
    {
      ++v14;
      v13 += 4;
      if ( v14 == v12 )
        return a1;
    }
  }
  v10 = *(unsigned __int16 *)(a1 + 2);
  v11 = 0;
  if ( a5 )
    v11 = a4;
  switch ( (__int16)v10 )
  {
    case '"':
      sub_BB52D0(&v35, a1);
      v16 = a3 - 1;
      v17 = *(unsigned __int8 **)a2;
      v18 = (__int64 *)(a2 + 8);
      v19 = *(_BYTE *)(a1 + 1) >> 1;
      if ( !a6 )
      {
        v24 = *(_BYTE *)(a1 + 1) >> 1;
        v33 = v16;
        v22 = sub_BB5290(a1, a1, v18);
        v19 = v24;
        v18 = (__int64 *)(a2 + 8);
        v16 = v33;
        a6 = v22;
      }
      v44 = 0;
      if ( v39 )
      {
        v41 = v36;
        if ( v36 > 0x40 )
        {
          v23 = v19;
          v26 = v18;
          v28 = v16;
          sub_C43780(&v40, &v35);
          v19 = v23;
          v18 = v26;
          v16 = v28;
        }
        else
        {
          v40 = v35;
        }
        v43 = v38;
        if ( v38 > 0x40 )
        {
          v25 = v19;
          v27 = v18;
          v34 = v16;
          sub_C43780(&v42, &v37);
          v19 = v25;
          v18 = v27;
          v16 = v34;
        }
        else
        {
          v42 = v37;
        }
        v44 = 1;
      }
      result = sub_AD9FD0(a6, v17, v18, v16, v19, (__int64)&v40, (__int64)v11);
      if ( v44 )
      {
        v44 = 0;
        if ( v43 > 0x40 && v42 )
        {
          v31 = result;
          j_j___libc_free_0_0(v42);
          result = v31;
        }
        if ( v41 > 0x40 && v40 )
        {
          v32 = result;
          j_j___libc_free_0_0(v40);
          result = v32;
        }
      }
      if ( v39 )
      {
        v39 = 0;
        if ( v38 > 0x40 && v37 )
        {
          v29 = result;
          j_j___libc_free_0_0(v37);
          result = v29;
        }
        if ( v36 > 0x40 )
        {
          if ( v35 )
          {
            v30 = result;
            j_j___libc_free_0_0(v35);
            result = v30;
          }
        }
      }
      break;
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
    case '1':
    case '2':
      result = sub_ADAB70(v10, *(_QWORD *)a2, a4, a5);
      break;
    case '=':
      result = sub_AD5840(*(_QWORD *)a2, *(unsigned __int8 **)(a2 + 8), (__int64)v11);
      break;
    case '>':
      result = sub_AD5A90(*(_QWORD *)a2, *(_BYTE **)(a2 + 8), *(unsigned __int8 **)(a2 + 16), (__int64)v11);
      break;
    case '?':
      v20 = (_DWORD *)sub_AC35F0(a1);
      result = sub_AD5CE0(*(_QWORD *)a2, *(_QWORD *)(a2 + 8), v20, v21, v11);
      break;
    default:
      result = sub_AD5570(v10, *(_QWORD *)a2, *(unsigned __int8 **)(a2 + 8), *(_BYTE *)(a1 + 1) >> 1, (__int64)v11);
      break;
  }
  return result;
}
