// Function: sub_DDD640
// Address: 0xddd640
//
__int64 __fastcall sub_DDD640(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r11d
  __int64 v7; // rbx
  __int64 v10; // r9
  bool v11; // al
  int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned __int8 v16; // r11
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // [rsp-C8h] [rbp-C8h]
  __int64 v21; // [rsp-C8h] [rbp-C8h]
  __int64 v22; // [rsp-C0h] [rbp-C0h]
  bool v23; // [rsp-C0h] [rbp-C0h]
  __int64 v24; // [rsp-C0h] [rbp-C0h]
  unsigned int v26; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v27; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v28; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v29; // [rsp-B0h] [rbp-B0h]
  __int64 v30; // [rsp-A8h] [rbp-A8h] BYREF
  int v31; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v32; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v33; // [rsp-90h] [rbp-90h]
  unsigned __int64 v34; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v35; // [rsp-80h] [rbp-80h]
  const void *v36; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v37; // [rsp-70h] [rbp-70h]
  unsigned __int8 v38; // [rsp-68h] [rbp-68h]
  const void *v39[2]; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int8 v40; // [rsp-48h] [rbp-48h]

  v6 = 0;
  if ( (((_DWORD)a2 - 36) & 0xFFFFFFFB) == 0 && *(_WORD *)(a3 + 24) == 8 && *(_WORD *)(a5 + 24) == 8 )
  {
    v7 = *(_QWORD *)(a5 + 48);
    if ( v7 != *(_QWORD *)(a3 + 48) )
      return v6;
    sub_DC06D0((__int64)&v36, (__int64)a1, a3, a5);
    v6 = v38;
    if ( !v38 )
      return v6;
    sub_DC06D0((__int64)v39, (__int64)a1, a4, a6);
    v6 = v40;
    if ( !v40 )
    {
LABEL_17:
      if ( v38 )
      {
        v28 = v6;
        v38 = 0;
        sub_969240((__int64 *)&v36);
        return v28;
      }
      return v6;
    }
    v10 = a6;
    if ( v37 <= 0x40 )
    {
      if ( v36 == v39[0] )
      {
        if ( v36 )
          goto LABEL_12;
      }
      else
      {
        LOBYTE(v6) = 0;
      }
    }
    else
    {
      v22 = a6;
      v26 = v37;
      v11 = sub_C43C50((__int64)&v36, v39);
      LOBYTE(v6) = v11;
      if ( v11 )
      {
        v20 = v22;
        v23 = v11;
        v12 = sub_C444A0((__int64)&v36);
        LOBYTE(v6) = v23;
        v10 = v20;
        if ( v26 != v12 )
        {
LABEL_12:
          v31 = 1;
          v30 = 0;
          if ( (_DWORD)a2 == 36 )
          {
            v24 = v10;
            sub_9865C0((__int64)&v32, (__int64)v39);
            v17 = v24;
            if ( v33 > 0x40 )
            {
              sub_C43D10((__int64)&v32);
              v17 = v24;
            }
            else
            {
              v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v33;
              if ( !v33 )
                v18 = 0;
              v32 = v18 & ~v32;
            }
            v21 = v17;
            sub_C46250((__int64)&v32);
          }
          else
          {
            v21 = v10;
            v13 = sub_D95540(a4);
            v14 = sub_D97050((__int64)a1, v13);
            sub_986680((__int64)&v32, v14);
            sub_C46B40((__int64)&v32, (__int64 *)v39);
          }
          v15 = v33;
          v33 = 0;
          v35 = v15;
          v34 = v32;
          sub_D91810(&v30, (__int64 *)&v34);
          sub_969240((__int64 *)&v34);
          sub_969240((__int64 *)&v32);
          v16 = sub_DAEB70((__int64)a1, v21, v7);
          if ( v16 )
          {
            v19 = sub_DA26C0(a1, (__int64)&v30);
            v16 = sub_DDD5B0(a1, v7, a2, v21, (__int64)v19);
          }
          v27 = v16;
          sub_969240(&v30);
          v6 = v27;
          if ( !v40 )
            goto LABEL_17;
        }
      }
    }
    v29 = v6;
    v40 = 0;
    sub_969240((__int64 *)v39);
    v6 = v29;
    goto LABEL_17;
  }
  return 0;
}
