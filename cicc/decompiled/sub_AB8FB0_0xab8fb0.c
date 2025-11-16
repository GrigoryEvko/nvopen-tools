// Function: sub_AB8FB0
// Address: 0xab8fb0
//
__int64 __fastcall sub_AB8FB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v5; // rax
  __int64 *v6; // r8
  unsigned __int64 v7; // r15
  unsigned int v8; // ebx
  unsigned __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // r8
  __int64 v12; // rax
  __int64 *v13; // r8
  unsigned __int64 v14; // rdx
  unsigned __int64 *v15; // rax
  int v16; // eax
  int v17; // eax
  unsigned int v18; // ecx
  unsigned __int64 *v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // eax
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // eax
  unsigned __int64 v26; // [rsp+0h] [rbp-B0h]
  __int64 *v27; // [rsp+8h] [rbp-A8h]
  unsigned int v28; // [rsp+8h] [rbp-A8h]
  __int64 *v29; // [rsp+10h] [rbp-A0h]
  unsigned int v30; // [rsp+10h] [rbp-A0h]
  unsigned int v31; // [rsp+10h] [rbp-A0h]
  unsigned int v32; // [rsp+10h] [rbp-A0h]
  __int64 *v33; // [rsp+10h] [rbp-A0h]
  __int64 *v34; // [rsp+18h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+28h] [rbp-88h]
  __int64 v37; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-78h]
  __int64 v39[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v40; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+58h] [rbp-58h]
  __int64 v42; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-48h]
  __int64 v44; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0((__int64)a3) )
  {
    sub_AB0A00((__int64)&v35, a2);
    sub_AB0910((__int64)&v37, a2);
    v5 = sub_9876C0(a3);
    v6 = v5;
    if ( !v5 )
    {
      sub_AB0910((__int64)v39, (__int64)a3);
      if ( (unsigned __int8)sub_AB06D0(a2) && (v20 = sub_9871D0((__int64)&v35), !sub_AAD8D0((__int64)v39, v20)) )
      {
        sub_AB0A00((__int64)&v44, (__int64)a3);
        sub_C47AC0(&v37, &v44);
        sub_969240(&v44);
        sub_C47AC0(&v35, v39);
        v24 = v38;
        v38 = 0;
        v41 = v24;
        v40 = v37;
        sub_C46A40(&v40, 1);
        v43 = v41;
        v41 = 0;
        v42 = v40;
        v25 = v36;
        v36 = 0;
        v45 = v25;
        v44 = v35;
        sub_9875E0(a1, &v44, &v42);
        sub_969240(&v44);
        sub_969240(&v42);
        sub_969240(&v40);
      }
      else
      {
        v21 = sub_9871A0((__int64)&v37);
        if ( sub_AAD8D0((__int64)v39, v21) )
        {
          sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
        }
        else
        {
          sub_AB0A00((__int64)&v44, (__int64)a3);
          sub_C47AC0(&v35, &v44);
          sub_969240(&v44);
          sub_C47AC0(&v37, v39);
          v22 = v38;
          v38 = 0;
          v41 = v22;
          v40 = v37;
          sub_C46A40(&v40, 1);
          v43 = v41;
          v41 = 0;
          v42 = v40;
          v23 = v36;
          v36 = 0;
          v45 = v23;
          v44 = v35;
          sub_9875E0(a1, &v44, &v42);
          sub_969240(&v44);
          sub_969240(&v42);
          sub_969240(&v40);
        }
      }
      sub_969240(v39);
      goto LABEL_21;
    }
    v7 = *(unsigned int *)(a2 + 8);
    v8 = *((_DWORD *)v5 + 2);
    if ( v8 > 0x40 )
    {
      v33 = v5;
      v16 = sub_9871A0((__int64)v5);
      v6 = v33;
      if ( v8 - v16 > 0x40 )
        goto LABEL_20;
      v9 = *(_QWORD *)*v33;
    }
    else
    {
      v9 = *v5;
    }
    v29 = v6;
    if ( v7 > v9 )
    {
      sub_9865C0((__int64)&v42, (__int64)&v35);
      v10 = v43;
      v11 = v29;
      if ( v43 > 0x40 )
      {
        sub_C43C10(&v42, &v37);
        v12 = v42;
        v10 = v43;
        v11 = v29;
      }
      else
      {
        v12 = v37 ^ v42;
        v42 ^= v37;
      }
      v27 = v11;
      v45 = v10;
      v44 = v12;
      v43 = 0;
      v30 = sub_9871A0((__int64)&v44);
      sub_969240(&v44);
      sub_969240(&v42);
      v13 = v27;
      v14 = v30;
      v31 = *((_DWORD *)v27 + 2);
      if ( v31 > 0x40 )
      {
        v26 = v14;
        v17 = sub_C444A0(v27);
        v13 = v27;
        v18 = v31 - v17;
        v19 = (unsigned __int64 *)*v27;
        if ( v18 > 0x40 || v26 < *v19 )
        {
          v15 = (unsigned __int64 *)*v19;
          goto LABEL_13;
        }
      }
      else
      {
        v15 = (unsigned __int64 *)*v27;
        if ( v14 < *v27 )
        {
LABEL_13:
          v32 = (unsigned int)v15;
          v28 = (unsigned int)v15;
          sub_9691E0((__int64)&v42, v7, 0, 0, 0);
          if ( v32 != v43 )
          {
            if ( v32 > 0x3F || v43 > 0x40 )
              sub_C43C90(&v42, v28, v43);
            else
              v42 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v28 - (unsigned __int8)v43 + 64) << v28;
          }
          sub_C46A40(&v42, 1);
          v45 = v43;
          v43 = 0;
          v44 = v42;
          sub_9691E0((__int64)&v40, v7, 0, 0, 0);
          goto LABEL_18;
        }
      }
      v34 = v13;
      sub_9865C0((__int64)&v42, (__int64)&v37);
      sub_C47AC0(&v42, v34);
      sub_C46A40(&v42, 1);
      v45 = v43;
      v43 = 0;
      v44 = v42;
      sub_9865C0((__int64)&v40, (__int64)&v35);
      sub_C47AC0(&v40, v34);
LABEL_18:
      sub_9875E0(a1, &v40, &v44);
      sub_969240(&v40);
      sub_969240(&v44);
      sub_969240(&v42);
LABEL_21:
      sub_969240(&v37);
      sub_969240(&v35);
      return a1;
    }
LABEL_20:
    sub_AADB10(a1, v7, 0);
    goto LABEL_21;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
