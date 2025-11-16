// Function: sub_2E309C0
// Address: 0x2e309c0
//
__int64 __fastcall sub_2E309C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  int v9; // eax
  int v10; // edx
  __int64 v11; // r8
  __int64 v12; // r9
  char *v13; // r10
  size_t v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  char **v17; // rax
  char ***v18; // rax
  __int64 v19; // rdx
  char v20; // cl
  char ***v21; // rsi
  int v22; // ecx
  int v23; // eax
  __int64 v24; // rdi
  char v25; // dl
  char **v26; // rdi
  __int64 v27; // [rsp-108h] [rbp-108h]
  char *v28; // [rsp-100h] [rbp-100h]
  unsigned __int64 v29; // [rsp-F8h] [rbp-F8h]
  size_t v30; // [rsp-F0h] [rbp-F0h]
  char **v31; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v32; // [rsp-E0h] [rbp-E0h]
  unsigned __int64 v33; // [rsp-D8h] [rbp-D8h]
  _BYTE v34[8]; // [rsp-D0h] [rbp-D0h] BYREF
  char *v35; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned __int64 v36; // [rsp-C0h] [rbp-C0h]
  int v37; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v38; // [rsp-A8h] [rbp-A8h]
  char **v39; // [rsp-98h] [rbp-98h] BYREF
  __int64 v40; // [rsp-90h] [rbp-90h]
  char *v41; // [rsp-88h] [rbp-88h]
  __int16 v42; // [rsp-78h] [rbp-78h]
  char ***v43; // [rsp-68h] [rbp-68h] BYREF
  __int64 v44; // [rsp-60h] [rbp-60h]
  char ***v45; // [rsp-58h] [rbp-58h]
  __int64 v46; // [rsp-50h] [rbp-50h]
  __int16 v47; // [rsp-48h] [rbp-48h]

  result = *(_QWORD *)(a1 + 264);
  if ( !result )
  {
    v7 = *(_QWORD *)(a1 + 32);
    v8 = *(_QWORD *)(v7 + 24);
    if ( *(_DWORD *)(v7 + 588) <= 2u && *(_BYTE *)(a1 + 260) )
    {
      v32 = 0;
      v31 = (char **)v34;
      v9 = *(_DWORD *)(a1 + 252);
      v33 = 5;
      v10 = *(_DWORD *)(a1 + 256);
      if ( unk_501EB38 == v9 && unk_501EB3C == v10 )
      {
        qmemcpy(v34, ".cold", 5);
        v16 = 5;
        v17 = (char **)v34;
        v32 = 5;
      }
      else if ( v9 == (_DWORD)qword_501EB30 && HIDWORD(qword_501EB30) == v10 )
      {
        v17 = (char **)v34;
        qmemcpy(v34, ".eh", 3);
        v16 = 3;
        v32 = 3;
      }
      else
      {
        LODWORD(v45) = *(_DWORD *)(a1 + 256);
        v41 = ".__part.";
        v42 = 773;
        v39 = (char **)v34;
        v40 = 0;
        v43 = &v39;
        v47 = 2306;
        sub_CA0F50((__int64 *)&v35, (void **)&v43);
        v13 = v35;
        v14 = v36;
        v15 = 0;
        v32 = 0;
        if ( v36 > v33 )
        {
          v28 = v35;
          v29 = v36;
          sub_C8D290((__int64)&v31, v34, v36, 1u, v11, v12);
          v15 = v32;
          v13 = v28;
          v14 = v29;
        }
        if ( v14 )
        {
          v30 = v14;
          memcpy((char *)v31 + v15, v13, v14);
          v15 = v32;
          v14 = v30;
        }
        v16 = v15 + v14;
        v32 = v16;
        if ( v35 != (char *)&v37 )
        {
          j_j___libc_free_0((unsigned __int64)v35);
          v16 = v32;
        }
        v17 = v31;
      }
      v39 = v17;
      v42 = 261;
      v40 = v16;
      v18 = (char ***)sub_2E791E0(v7);
      v20 = v42;
      if ( (_BYTE)v42 )
      {
        if ( (_BYTE)v42 == 1 )
        {
          v43 = v18;
          v44 = v19;
          v47 = 261;
        }
        else
        {
          if ( HIBYTE(v42) == 1 )
          {
            v27 = v40;
            v21 = (char ***)v39;
          }
          else
          {
            v21 = &v39;
            v20 = 2;
          }
          v43 = v18;
          v44 = v19;
          v45 = v21;
          v46 = v27;
          LOBYTE(v47) = 5;
          HIBYTE(v47) = v20;
        }
      }
      else
      {
        v47 = 256;
      }
      result = sub_E6C460(v8, (const char **)&v43);
      v26 = v31;
      *(_QWORD *)(a1 + 264) = result;
      if ( v26 != (char **)v34 )
      {
        _libc_free((unsigned __int64)v26);
        return *(_QWORD *)(a1 + 264);
      }
    }
    else
    {
      v22 = *(_DWORD *)(v7 + 336);
      v23 = *(_DWORD *)(a1 + 24);
      v24 = *(_QWORD *)(v7 + 24);
      v35 = "BB";
      v25 = *(_BYTE *)(a1 + 232);
      v41 = "_";
      v37 = v22;
      v39 = &v35;
      v38 = 2307;
      v42 = 770;
      v43 = &v39;
      LODWORD(v45) = v23;
      v47 = 2562;
      result = sub_E6C900(v24, (__int64 *)&v43, v25, (__int64)&v39, a5, 2307);
      *(_QWORD *)(a1 + 264) = result;
    }
  }
  return result;
}
