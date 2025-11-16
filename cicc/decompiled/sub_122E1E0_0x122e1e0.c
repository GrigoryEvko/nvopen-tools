// Function: sub_122E1E0
// Address: 0x122e1e0
//
__int64 __fastcall sub_122E1E0(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  __int64 v3; // r12
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp+0h] [rbp-50h] BYREF
  char v8; // [rsp+20h] [rbp-30h]
  char v9; // [rsp+21h] [rbp-2Fh]

  v3 = a1 + 248;
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "DILocation") )
    return sub_1225E90(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIExpression") )
    return sub_1210190(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIGlobalVariableExpression") )
    return sub_122B510(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "GenericDINode") )
    return sub_1225840(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DISubrange") )
    return sub_122C950((_QWORD **)a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIEnumerator") )
    return sub_120ECF0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIBasicType") )
    return sub_120F380(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIDerivedType") )
    return sub_1226670(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DICompositeType") )
    return sub_122D3C0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DISubroutineType") )
    return sub_1226F90(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIFile") )
    return sub_120F790(a1, (const __m128i **)a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DICompileUnit") )
    return sub_1227E40(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DISubprogram") )
    return sub_1227E90(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DILexicalBlock") )
    return sub_1228C20(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DILexicalBlockFile") )
    return sub_1228F60(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DINamespace") )
    return sub_12295F0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIModule") )
    return sub_1229BF0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DITemplateTypeParameter") )
    return sub_122A080(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DITemplateValueParameter") )
    return sub_122A340(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIGlobalVariable") )
    return sub_122A6C0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DILocalVariable") )
    return sub_122ACA0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DILabel") )
    return sub_122B170(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIObjCProperty") )
    return sub_122B7B0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIImportedEntity") )
    return sub_122BBB0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIAssignID") )
    return sub_120EC00(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIMacro") )
    return sub_120FDC0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIMacroFile") )
    return sub_12298B0(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DICommonBlock") )
    return sub_1229260(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIStringType") )
    return sub_1226210(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DIGenericSubrange") )
    return sub_122CE90(a1, a2, a3);
  if ( !(unsigned int)sub_2241AC0(v3, "DISubrangeType") )
    return sub_122C170((_QWORD **)a1, a2, a3);
  v6 = *(_QWORD *)(a1 + 232);
  v9 = 1;
  v7 = "expected metadata type";
  v8 = 3;
  sub_11FD800(a1 + 176, v6, (__int64)&v7, 1);
  return 1;
}
