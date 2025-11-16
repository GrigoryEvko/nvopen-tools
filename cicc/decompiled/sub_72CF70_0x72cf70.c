// Function: sub_72CF70
// Address: 0x72cf70
//
void sub_72CF70()
{
  __int64 v0; // r12
  _BYTE *v1; // rax
  __int64 v2; // r12
  _BYTE *v3; // rax
  __int64 v4; // r12
  _BYTE *v5; // rax
  __int64 v6; // r12
  _BYTE *v7; // rax
  __int64 v8; // r12
  _BYTE *v9; // rax
  __int64 v10; // r12
  _BYTE *v11; // rax
  __int64 v12; // r12
  _BYTE *v13; // rax
  __int64 v14; // r12
  _BYTE *v15; // rax
  __int64 v16; // r12
  _BYTE *v17; // rax
  __int64 v18; // r12
  _BYTE *v19; // rax

  if ( !unk_4F06C30 )
  {
    v0 = sub_72CC70();
    if ( (unsigned int)sub_8D3A70(v0) )
      v1 = sub_72CA00(v0, "equal");
    else
      v1 = sub_72C9A0();
    unk_4F06C30 = v1;
    v2 = sub_72CC70();
    if ( (unsigned int)sub_8D3A70(v2) )
      v3 = sub_72CA00(v2, "less");
    else
      v3 = sub_72C9A0();
    unk_4F06C28 = v3;
    v4 = sub_72CC70();
    if ( (unsigned int)sub_8D3A70(v4) )
      v5 = sub_72CA00(v4, "greater");
    else
      v5 = sub_72C9A0();
    unk_4F06C20 = v5;
    v6 = sub_72CCA0();
    if ( (unsigned int)sub_8D3A70(v6) )
      v7 = sub_72CA00(v6, "equivalent");
    else
      v7 = sub_72C9A0();
    unk_4F06C18 = v7;
    v8 = sub_72CCA0();
    if ( (unsigned int)sub_8D3A70(v8) )
      v9 = sub_72CA00(v8, "less");
    else
      v9 = sub_72C9A0();
    unk_4F06C10 = v9;
    v10 = sub_72CCA0();
    if ( (unsigned int)sub_8D3A70(v10) )
      v11 = sub_72CA00(v10, "greater");
    else
      v11 = sub_72C9A0();
    unk_4F06C08 = v11;
    v12 = sub_72CCD0();
    if ( (unsigned int)sub_8D3A70(v12) )
      v13 = sub_72CA00(v12, "equivalent");
    else
      v13 = sub_72C9A0();
    unk_4F06C00 = v13;
    v14 = sub_72CCD0();
    if ( (unsigned int)sub_8D3A70(v14) )
      v15 = sub_72CA00(v14, "less");
    else
      v15 = sub_72C9A0();
    unk_4F06BF8 = v15;
    v16 = sub_72CCD0();
    if ( (unsigned int)sub_8D3A70(v16) )
      v17 = sub_72CA00(v16, "greater");
    else
      v17 = sub_72C9A0();
    unk_4F06BF0 = v17;
    v18 = sub_72CCD0();
    if ( (unsigned int)sub_8D3A70(v18) )
      v19 = sub_72CA00(v18, "unordered");
    else
      v19 = sub_72C9A0();
    unk_4F06BE8 = v19;
  }
}
